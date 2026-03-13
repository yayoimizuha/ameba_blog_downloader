/// clean_non_jpeg: ディレクトリを再帰的に走査し、JPEG以外のファイルを削除、
/// JPEGはヘッダをパースして整合性を検証するツール。
use clap::Parser;
use std::path::{Path, PathBuf};
use tracing::{error, info, warn};

// ─────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "clean_non_jpeg",
    about = "Scan a directory recursively, delete non-JPEG files, and validate JPEG headers."
)]
struct Args {
    /// 走査対象ディレクトリ
    directory: PathBuf,

    /// このフラグを指定すると実際の削除は行わず、対象ファイルを表示するだけ
    #[arg(long, short = 'n')]
    dry_run: bool,
}

// ─────────────────────────────────────────────
// JPEG ヘッダ検証
// ─────────────────────────────────────────────

/// JPEG マーカー定数
const MARKER_PREFIX: u8 = 0xFF;
const MARKER_SOI: u8 = 0xD8; // Start of Image
const MARKER_EOI: u8 = 0xD9; // End of Image
const MARKER_SOS: u8 = 0xDA; // Start of Scan
/// 長さフィールドを持たないスタンドアロンマーカー群
const STANDALONE_MARKERS: &[u8] = &[
    0xD8, // SOI
    0xD9, // EOI
    0x01, // TEM
];
/// RST0〜RST7 (0xD0〜0xD7) もスタンドアロン
fn is_standalone(marker: u8) -> bool {
    STANDALONE_MARKERS.contains(&marker) || (0xD0..=0xD7).contains(&marker)
}

#[derive(Debug)]
enum JpegError {
    TooSmall,
    NoSoi,
    NoEoi,
    InvalidMarker { offset: usize },
    SegmentOutOfBounds { offset: usize },
    SegmentTooSmall { offset: usize },
    NoSos,
}

impl std::fmt::Display for JpegError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JpegError::TooSmall => write!(f, "ファイルが小さすぎます (< 4 bytes)"),
            JpegError::NoSoi => write!(f, "SOI マーカー (FF D8) が先頭にありません"),
            JpegError::NoEoi => write!(f, "EOI マーカー (FF D9) が末尾にありません"),
            JpegError::InvalidMarker { offset } => {
                write!(f, "不正なマーカー: offset=0x{offset:08X}")
            }
            JpegError::SegmentOutOfBounds { offset } => {
                write!(f, "セグメント長がファイル境界を超えています: offset=0x{offset:08X}")
            }
            JpegError::SegmentTooSmall { offset } => {
                write!(
                    f,
                    "セグメント長フィールドが 2 未満です: offset=0x{offset:08X}"
                )
            }
            JpegError::NoSos => write!(f, "SOS マーカー (FF DA) が見つかりません"),
        }
    }
}

/// バイト列を受け取り JPEG ヘッダ構造を検証する。
/// SOS に到達した時点でスキャンデータは検証しない（高速化のため）。
fn validate_jpeg(data: &[u8]) -> Result<(), JpegError> {
    // 最低限 SOI (2) + 何らかのセグメント (4) + EOI (2) で 4 bytes は必要
    if data.len() < 4 {
        return Err(JpegError::TooSmall);
    }

    // SOI 確認
    if data[0] != MARKER_PREFIX || data[1] != MARKER_SOI {
        return Err(JpegError::NoSoi);
    }

    // EOI 確認（末尾 2 bytes）
    let tail = data.len() - 2;
    if data[tail] != MARKER_PREFIX || data[tail + 1] != MARKER_EOI {
        return Err(JpegError::NoEoi);
    }

    // セグメントを順次パース
    let mut pos = 2usize; // SOI の直後から
    let mut found_sos = false;

    while pos < data.len() {
        // マーカープレフィックス確認
        if data[pos] != MARKER_PREFIX {
            return Err(JpegError::InvalidMarker { offset: pos });
        }

        // 0xFF のパディングをスキップ
        let mut marker_byte_pos = pos + 1;
        while marker_byte_pos < data.len() && data[marker_byte_pos] == MARKER_PREFIX {
            marker_byte_pos += 1;
        }

        if marker_byte_pos >= data.len() {
            return Err(JpegError::InvalidMarker { offset: pos });
        }

        let marker = data[marker_byte_pos];
        let segment_start = marker_byte_pos + 1; // 長さフィールドの先頭

        if marker == MARKER_EOI {
            // EOI に到達 → 正常終了
            break;
        }

        if marker == MARKER_SOS {
            found_sos = true;
            // SOS 以降はエントロピー符号化データのため解析を終了
            break;
        }

        if is_standalone(marker) {
            // 長さフィールドなし → 次の 0xFF へ
            pos = segment_start;
            continue;
        }

        // 長さフィールドを読む (big-endian, 長さ自身の 2 bytes を含む)
        if segment_start + 1 >= data.len() {
            return Err(JpegError::SegmentOutOfBounds {
                offset: segment_start,
            });
        }
        let length = u16::from_be_bytes([data[segment_start], data[segment_start + 1]]) as usize;
        if length < 2 {
            return Err(JpegError::SegmentTooSmall {
                offset: segment_start,
            });
        }

        let next = segment_start + length;
        if next > data.len() {
            return Err(JpegError::SegmentOutOfBounds {
                offset: segment_start,
            });
        }

        pos = next;
    }

    if !found_sos {
        return Err(JpegError::NoSos);
    }

    Ok(())
}

// ─────────────────────────────────────────────
// ファイル判定ユーティリティ
// ─────────────────────────────────────────────

fn is_jpeg_extension(path: &Path) -> bool {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("jpg") | Some("jpeg") => true,
        _ => false,
    }
}

// ─────────────────────────────────────────────
// ディレクトリ走査
// ─────────────────────────────────────────────

#[derive(Default)]
struct Stats {
    jpeg_ok: usize,
    jpeg_warn: usize,
    non_jpeg_deleted: usize,
    non_jpeg_skipped: usize, // dry-run 時
    errors: usize,
}

fn process_dir(dir: &Path, dry_run: bool, stats: &mut Stats) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            error!("ディレクトリを開けません: {} — {}", dir.display(), e);
            stats.errors += 1;
            return;
        }
    };

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                error!("エントリの読み取りエラー: {}", e);
                stats.errors += 1;
                continue;
            }
        };

        let path = entry.path();
        let file_type = match entry.file_type() {
            Ok(ft) => ft,
            Err(e) => {
                error!("ファイル種別取得エラー: {} — {}", path.display(), e);
                stats.errors += 1;
                continue;
            }
        };

        if file_type.is_dir() {
            process_dir(&path, dry_run, stats);
        } else if file_type.is_file() {
            process_file(&path, dry_run, stats);
        }
        // シンボリックリンク等は無視
    }
}

fn process_file(path: &Path, dry_run: bool, stats: &mut Stats) {
    if is_jpeg_extension(path) {
        // JPEG: ヘッダ検証
        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                error!("[READ ERROR] {} — {}", path.display(), e);
                stats.errors += 1;
                return;
            }
        };
        match validate_jpeg(&data) {
            Ok(()) => {
                info!("[JPEG OK]   {}", path.display());
                stats.jpeg_ok += 1;
            }
            Err(e) => {
                warn!("[JPEG WARN] {} — {}", path.display(), e);
                stats.jpeg_warn += 1;
            }
        }
    } else {
        // 非 JPEG: 削除
        if dry_run {
            info!("[DRY-RUN DELETE] {}", path.display());
            stats.non_jpeg_skipped += 1;
        } else {
            match std::fs::remove_file(path) {
                Ok(()) => {
                    info!("[DELETED]   {}", path.display());
                    stats.non_jpeg_deleted += 1;
                }
                Err(e) => {
                    error!("[DELETE ERROR] {} — {}", path.display(), e);
                    stats.errors += 1;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────
// エントリポイント
// ─────────────────────────────────────────────

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    if !args.directory.is_dir() {
        eprintln!(
            "エラー: '{}' はディレクトリではありません",
            args.directory.display()
        );
        std::process::exit(1);
    }

    if args.dry_run {
        info!("=== DRY-RUN モード: ファイルは削除されません ===");
    }

    info!("走査開始: {}", args.directory.display());

    let mut stats = Stats::default();
    process_dir(&args.directory, args.dry_run, &mut stats);

    // ─── サマリー ───
    println!("\n========== 処理結果サマリー ==========");
    println!("  JPEG 正常           : {:>6}", stats.jpeg_ok);
    println!("  JPEG 問題あり (警告) : {:>6}", stats.jpeg_warn);
    if args.dry_run {
        println!(
            "  非JPEG (削除予定)    : {:>6}  ※ dry-run のため削除せず",
            stats.non_jpeg_skipped
        );
    } else {
        println!("  非JPEG 削除済み      : {:>6}", stats.non_jpeg_deleted);
    }
    println!("  エラー              : {:>6}", stats.errors);
    println!("======================================");
}
