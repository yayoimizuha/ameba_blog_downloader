// use itertools::Itertools;
// use std::any::type_name;
use ameba_blog_downloader::data_dir;
use itertools::Itertools;
use std::fs;
use std::time::Instant;
use zerocopy::IntoBytes;

// static CHECK_ARRAY: fn(&[u8], &[u8]) -> bool = |right: &[u8], left: &[u8]| right.iter().zip_eq(left).all(|(a, b)| a == b);

#[allow(unused_macros)]
macro_rules! assert_bytes {
    ($ptr:expr,$arr:expr,$literal:expr) => {{
        let literal_len: usize = $literal.len();
        let literal_vec = (0..literal_len).step_by(2).map(|i| u8::from_str_radix(&$literal[i..i + 2], 16).unwrap()).collect::<Vec<u8>>();
        let literal_slice = literal_vec.as_slice();
        $arr[$ptr..($ptr + literal_len / 2 as usize)].iter().zip_eq(literal_slice).all(|(a, b)| a == b)
    }};
}

#[derive(Debug)]
#[allow(dead_code)]
enum IfdDtype {
    U8(Vec<u8>),
    ASCII(String),
    SHORT(Vec<u16>),
    LONG(Vec<u32>),
    RATIONAL(Vec<(u32, u32)>),
    I8(Vec<i8>),
    UNDEFINED(Vec<u8>),
    SSHORT(Vec<i16>),
    SLONG(Vec<i32>),
    SRATIONAL(Vec<(i32, i32)>),
}

macro_rules! repr_ifd_dtype {
    ($data_type:ty,$data_size:expr,$val_or_offset:expr,$later:expr) => {{
        // println!("sizeof::<{}>()={}",type_name::<$data_type>(),size_of::<$data_type>());
        if size_of::<$data_type>() * $data_size < 4 {
            &$val_or_offset[0..size_of::<$data_type>() * $data_size as usize]
        } else {
            let offset = u32::from_be_bytes(<[u8; 4]>::try_from($val_or_offset).unwrap()) as usize;
            &$later[offset..offset + (size_of::<$data_type>() * $data_size as usize)]
        }
    }};
}
macro_rules! infer_primitive_type {
    ($data_type:ty,$data_size:expr,$val_or_offset:expr,$later:expr) => {
        repr_ifd_dtype!($data_type, $data_size, $val_or_offset, $later)
            .chunks(size_of::<$data_type>())
            .map(|v| <$data_type>::from_be_bytes(<[u8; size_of::<$data_type>()]>::try_from(v).unwrap()))
            .collect::<Vec<_>>()
    };
}
macro_rules! infer_rational_type {
    ($data_type:ty,$data_size:expr,$val_or_offset:expr,$later:expr) => {
        repr_ifd_dtype!(u32, $data_size * 2, $val_or_offset, $later)
            .chunks(size_of::<$data_type>() * 2)
            .map(|v| {
                let numerator = <$data_type>::from_be_bytes(<[u8; size_of::<$data_type>()]>::try_from(&v[..4]).unwrap());
                let denominator = <$data_type>::from_be_bytes(<[u8; size_of::<$data_type>()]>::try_from(&v[4..]).unwrap());
                (numerator, denominator)
            })
            .collect::<Vec<_>>()
    };
}
impl IfdDtype {
    fn new(data_type: u16, data_size: u16, val_or_offset: &[u8], later: &[u8]) -> IfdDtype {
        let data_size = data_size as usize;
        match data_type {
            1 => IfdDtype::U8(Vec::from(repr_ifd_dtype!(u8, data_size, val_or_offset, later))),
            2 => IfdDtype::ASCII(String::from_utf8((&repr_ifd_dtype!(u8, data_size, val_or_offset, later)).to_vec()).unwrap().as_str().strip_suffix("\0").unwrap().to_string()),
            3 => IfdDtype::SHORT(infer_primitive_type!(u16, data_size, val_or_offset, later)),
            4 => IfdDtype::LONG(infer_primitive_type!(u32, data_size, val_or_offset, later)),
            5 => IfdDtype::RATIONAL(infer_rational_type!(u32, data_size, val_or_offset, later)),
            6 => IfdDtype::I8(Vec::from(repr_ifd_dtype!(i8, data_size, val_or_offset, later)).into_iter().map(|v| v as i8).collect()),
            7 => IfdDtype::UNDEFINED(Vec::from(repr_ifd_dtype!(u8, data_size, val_or_offset, later))),
            8 => IfdDtype::SSHORT(infer_primitive_type!(i16, data_size, val_or_offset, later)),
            9 => IfdDtype::SLONG(infer_primitive_type!(i32, data_size, val_or_offset, later)),
            10 => IfdDtype::SRATIONAL(infer_rational_type!(i32, data_size, val_or_offset, later)),
            _ => unimplemented!(),
        }
    }
}

fn main() {
    // let file = include_bytes!("test.jpg");
    // parse_jpeg(include_bytes!("test.jpg"));
    let image_dir = data_dir().join("blog_images").read_dir().unwrap().map(|dir| dir.unwrap().path().read_dir().unwrap().into_iter().map(|file| file.unwrap().path())).flatten().collect::<Vec<_>>();
    let start = Instant::now();
    let _ = image_dir
        .into_iter()
        .take(1)
        .map(|file| {
            println!("{}", file.to_str().unwrap());
            parse_jpeg(&*fs::read(file).unwrap())
        })
        .collect::<Vec<_>>();
    println!("{:?}", Instant::now() - start)
    // println!("size_of::<u8>()={}", size_of::<u8>());
    // for word in file.chunks(8) {
    //     for byte in word {
    //         print!("{:02X} ", byte);
    //     }
    //     println!();
    //     break;
    // }
    //
    // assert_bytes!(ptr, file, "FFD8");
    // ptr += 2;
    //
    // println!("{}", u16::from_be_bytes(<[u8; 2]>::try_from(&file[ptr..ptr + 2]).unwrap()));
}
static LOG_LEVEL: i64 = 5; // 2~5
fn parse_jpeg(file: &[u8]) {
    let mut ptr = 0;
    #[allow(unused_assignments)]
    let mut restart_interval = None;
    macro_rules! print_indent {
    () => {
        std::print!("\n")
    };
    ($indent:expr,$log_level:expr,$($arg:tt)*) => {{
        if $log_level < LOG_LEVEL {
        print!("{}",(0..$indent*4).map(|_| " ").join(""));
        println!($($arg)*);
        }
    }};
}

    loop {
        match file[ptr..].as_bytes() {
            [0xFF, 0xD8, ..] => {
                print_indent!(0, 1, "SOI (Start Of Image [FF D8]) detected. @ {:#08X}", ptr);
                ptr += 2;
            }
            [0xFF, 0xD9, ..] => {
                print_indent!(0, 1, "EOI (End Of Image [FF D9]) detected. @ {:#08X}", ptr);
                break;
            }
            [0xFF, 0xDD, later @ ..] => {
                print_indent!(0, 1, "DRI (Define Restart Interval [FF DD]) detected. @ {:#08X}", ptr);
                let segment_size = u16::from_be_bytes(<[u8; 2]>::try_from(&later[..2]).unwrap());
                print_indent!(1, 2, "segment size: {}", segment_size);
                restart_interval = Some(u16::from_be_bytes(<[u8; 2]>::try_from(&later[2..4]).unwrap()));
                print_indent!(1, 2, "restart interval: {}", restart_interval.unwrap());
                ptr += 2 + segment_size as usize;
            }
            [0xFF, 0xC0, later @ ..] => {
                print_indent!(0, 1, "SOF (Start Of Frame type 0(baseline) [FF C0]) detected. @ {:#08X}", ptr);
                let segment_size = u16::from_be_bytes(<[u8; 2]>::try_from(&later[..2]).unwrap());
                print_indent!(1, 2, "segment size: {}", segment_size);
                let sample_precision = later[2];
                print_indent!(1, 2, "sample precision: {}", sample_precision);
                let image_height = u16::from_be_bytes(<[u8; 2]>::try_from(&later[3..=4]).unwrap());
                let image_width = u16::from_be_bytes(<[u8; 2]>::try_from(&later[5..=6]).unwrap());
                print_indent!(1, 2, "image size: HxW: {image_height}x{image_width}");
                let number_of_components = later[7];
                print_indent!(1, 2, "number of components: {}", number_of_components);
                for i in 0..number_of_components {
                    print_indent!(
                        2,
                        3,
                        "component ID: {}={}",
                        later[8 + i as usize * 3],
                        match later[8 + i as usize * 3] {
                            1 => "Y",
                            2 => "Cb",
                            3 => "Cr",
                            4 => "I",
                            5 => "Q",
                            _ => unreachable!(),
                        }
                    );
                    print_indent!(3, 3, "Horizontal sampling factor: {}", later[9 + i as usize * 3] >> 4);
                    print_indent!(3, 3, "Vertical sampling factor: {}", later[9 + i as usize * 3] & 0x0F);
                    print_indent!(3, 3, "Quantization table destination selector: {}", later[10 + i as usize * 3] & 0x0F);
                }
                ptr += 2 + segment_size as usize;
            }
            [0xFF, 0xE0, later @ ..] => {
                print_indent!(0, 1, "APP0 (Application type 0 segment [FF E0]) detected. @ {:#08X}", ptr);
                let segment_size = u16::from_be_bytes(<[u8; 2]>::try_from(&later[..2]).unwrap());
                print_indent!(1, 2, "segment size: {}", segment_size);
                assert_eq!(later[2..7], *"JFIF\0".as_bytes());
                print_indent!(1, 2, "JFIF Ver: {:x}.{:02x}", later[7], later[8]);
                let scale_unit = later[9];
                print_indent!(
                    1,
                    2,
                    "scale unit :{}={}",
                    scale_unit,
                    match scale_unit {
                        0 => "undefined",
                        1 => "dots per inch",
                        2 => "dots per cm",
                        _ => unreachable!(),
                    }
                );
                let pixel_width = u16::from_be_bytes(<[u8; 2]>::try_from(&later[10..=11]).unwrap());
                let pixel_height = u16::from_be_bytes(<[u8; 2]>::try_from(&later[12..=13]).unwrap());
                print_indent!(1, 2, "HxW : {}x{} per unit", pixel_height, pixel_width);
                let thumb_width = later[14];
                let thumb_height = later[15];
                print_indent!(1, 2, "thumbnail HxW : {}x{} px", thumb_height, thumb_width);
                later[16..segment_size as usize]
                    .chunks(8)
                    .map(|r| {
                        r.iter().map(|v| print!("{v:02X} ")).count();
                        println!();
                    })
                    .count();
                ptr += 2 + segment_size as usize;
            }
            [0xFF, 0xE1, later @ ..] => {
                // http://www.ryouto.jp/f6exif/exif.html
                print_indent!(0, 1, "APP1 (Application type 1 segment [FF E1])(Exif) detected. @ {:#08X}", ptr);
                let segment_size = u16::from_be_bytes(<[u8; 2]>::try_from(&later[..2]).unwrap());
                print_indent!(1, 2, "segment size: {}", segment_size);
                assert_eq!(later[2..8], *"Exif\0\0".as_bytes());
                assert_eq!(later[8..10], *"MM".as_bytes());
                assert_eq!(u16::from_be_bytes(<[u8; 2]>::try_from(&later[10..12]).unwrap()), 0x002A);
                let mut ifd_offset = u32::from_be_bytes(<[u8; 4]>::try_from(&later[12..16]).unwrap()) as usize;
                loop {
                    // loop IFD
                    let ifd_block = &later[ifd_offset + 8..];
                    let entry_count = u16::from_be_bytes(<[u8; 2]>::try_from(&ifd_block[..2]).unwrap()) as usize;
                    print_indent!(2, 3, "new IFD found.");
                    for i in 0..entry_count {
                        let entry = &ifd_block[(2 + i * 12)..(14 + i * 12)];
                        let tag = u16::from_be_bytes(<[u8; 2]>::try_from(&entry[..2]).unwrap());
                        let data_type = u16::from_be_bytes(<[u8; 2]>::try_from(&entry[2..4]).unwrap());
                        let data_size = u16::from_be_bytes(<[u8; 2]>::try_from(&entry[6..8]).unwrap());
                        let val_or_offset = &entry[8..];
                        // println!("\t\ttag: {tag:#04X}\tdata_type: {data_type}\tdata_size: {data_size}\tval_or_offset: {}", val_or_offset.into_iter().map(|v| format!("{v:02X} ")).join(""));
                        print_indent!(3, 4, "{:#04X}:{:?}", tag, IfdDtype::new(data_type, data_size, &val_or_offset, &later[8..]));
                    }
                    ifd_offset = u32::from_be_bytes(<[u8; 4]>::try_from(&ifd_block[2 + entry_count * 12..6 + entry_count * 12]).unwrap()) as usize;
                    if ifd_offset == 0 {
                        break;
                    }
                }
                ptr += 2 + segment_size as usize;
            }
            [0xFF, 0xDA, later @ ..] => {
                print_indent!(0, 1, "SOS (Start Of Scan segment [FF DA]) detected. @ {:#08X}", ptr);
                let segment_size = u16::from_be_bytes(<[u8; 2]>::try_from(&later[..2]).unwrap());
                print_indent!(1, 2, "segment size: {}", segment_size);
                let component_count = later[2];
                print_indent!(1, 2, "number of components: {}", component_count);
                for i in 0..component_count {
                    print_indent!(
                        2,
                        3,
                        "component ID: {}={}",
                        later[3 + i as usize * 2],
                        match later[3 + i as usize * 2] {
                            1 => "Y",
                            2 => "Cb",
                            3 => "Cr",
                            4 => "I",
                            5 => "Q",
                            _ => unreachable!(),
                        }
                    );
                    print_indent!(3, 3, "DC Huffman table Number: {}", later[4 + i as usize * 2] >> 4);
                    print_indent!(3, 3, "AC Huffman table Number: {}", later[4 + i as usize * 2] & 0x0F);
                }
                let start_of_spectral_selection = later[3 + component_count as usize * 2];
                print_indent!(1, 2, "start of spectral selection: {start_of_spectral_selection}");
                let end_of_spectral_selection = later[4 + component_count as usize * 2];
                print_indent!(1, 2, "end of spectral selection: {end_of_spectral_selection}");
                let approx_spectral_shift = later[5 + component_count as usize * 2] >> 4;
                print_indent!(1, 2, "approximation spectral shift: {approx_spectral_shift}");
                let spectral_shift = later[5 + component_count as usize * 2] & 0xFF;
                print_indent!(1, 2, "spectral shift: {spectral_shift}");
                // ptr += 2 + segment_size as usize;
                let mcus = [later[segment_size as usize..].windows(2).filter_map(|window| {
                    let (a, b) = (window[0], window[1]);
                    match a == 0xFF {
                        true => {
                            match b {
                                0x00 => Some(0xFF),
                                _rst_id @ 0xD0..0xD8 => {
                                    println!("\tRST{} marker detected. skip...", _rst_id - 0xD0);
                                    None
                                }
                                0xD9 => {
                                    println!("EOI (End of Image [FF D9]) detected.");
                                    None
                                },
                                _ => panic!()
                            }
                        }
                        false => { Some(a) }
                    }
                }).collect::<Vec<_>>(), vec![*later.last().unwrap()]].concat();
                ptr +=  later.len();
                // for pos in segment_size as usize..later.len() {
                //     if later[pos] == 0xFF {
                //         match later[pos + 1] {
                //             0x00 => {}
                //             _rst_id @ 0xD0..0xD8 => {
                //                 println!("\tRST{} marker @ {:#08X} detected. skip...", _rst_id - 0xD0, ptr + 2 + pos);
                //             }
                //             0xD9 => {
                //                 break
                //             }
                //             _ => {
                //                 ptr += 2 + pos;
                //                 break;
                //             }
                //         }
                //     }
                // }
            }
            [0xFF, app_id @ 0xE0..=0xEF, later @ ..] => {
                print_indent!(0, 1, "APP{0} (Application type {0} segment [FF {app_id:02X}]) detected. @ {1:#08X}", app_id - 0xE0, ptr);
                let segment_size = u16::from_be_bytes(<[u8; 2]>::try_from(&later[..2]).unwrap());
                print_indent!(1, 2, "segment size: {}", segment_size);
                ptr += 2 + segment_size as usize;
            }
            [0xFF, 0xDB, later @ ..] => {
                print_indent!(0, 1, "DQT (Define quantization table segment [FF DB]) detected. @ {:#08X}", ptr);
                let segment_size = u16::from_be_bytes(<[u8; 2]>::try_from(&later[..2]).unwrap());
                print_indent!(1, 2, "segment size: {}", segment_size);
                let mut dqt_ptr = 2;
                loop {
                    let quantize_precision = (later[dqt_ptr] & 0xF0) >> 4;
                    print_indent!(2, 3, "quantize precision: {}", quantize_precision);
                    let quantize_table_identifier = later[dqt_ptr] & 0x0F;
                    print_indent!(2, 3, "quantize table id: {}", quantize_table_identifier);
                    dqt_ptr += 1;
                    for _ in 0..8 {
                        if LOG_LEVEL > 4 {
                            print!("\t    ");
                        }
                        for _ in 0..8 {
                            let val = match quantize_precision {
                                0 => {
                                    let v = later[dqt_ptr];
                                    dqt_ptr += 1;
                                    v as u16
                                }
                                1 => {
                                    let v = u16::from_be_bytes(<[u8; 2]>::try_from(&later[dqt_ptr..dqt_ptr + 2]).unwrap());
                                    dqt_ptr += 2;
                                    v
                                }
                                _ => unreachable!(),
                            };
                            if LOG_LEVEL > 4 {
                                print!("{val:02X} ");
                            }
                        }
                        if LOG_LEVEL > 4 {
                            println!();
                        }
                    }
                    if segment_size as usize == dqt_ptr {
                        break;
                    }
                }
                ptr += 2 + segment_size as usize;
            }
            [0xFF, 0xC4, later @ ..] => {
                print_indent!(0, 1, "DHT (Define Huffman table segment [FF C4]) detected. @ {:#08X}", ptr);
                let segment_size = u16::from_be_bytes(<[u8; 2]>::try_from(&later[..2]).unwrap());
                print_indent!(1, 2, "segment size: {}", segment_size);
                let mut dht_ptr = 2;
                loop {
                    let huffman_table_class = later[dht_ptr] >> 4;
                    print_indent!(
                        2,
                        3,
                        "huffman table class: {}(={})",
                        huffman_table_class,
                        match huffman_table_class {
                            0 => "DC",
                            1 => "AC",
                            _ => unreachable!(),
                        }
                    );
                    let huffman_table_id = later[dht_ptr] & 0x0F;
                    print_indent!(2, 3, "huffman table destination id: {}", huffman_table_id);
                    print_indent!(2, 4, "code counts per code length:");
                    dht_ptr += 17;
                    for (bits, code_counts) in later[dht_ptr - 16..dht_ptr].into_iter().enumerate() {
                        if *code_counts == 0 {
                            continue;
                        }
                        if LOG_LEVEL > 4 {
                            print!("\t    codes of {bits:2>} bit(s):");
                        }
                        for _ in 0..*code_counts {
                            if LOG_LEVEL > 4 {
                                print!(" {:02X}", later[dht_ptr]);
                            }
                            dht_ptr += 1;
                        }
                        if LOG_LEVEL > 4 {
                            println!();
                        }
                    }
                    if segment_size as usize == dht_ptr {
                        break;
                    }
                }
                ptr += 2 + segment_size as usize;
            }
            _ => {
                print_indent!(0, 1, "Unknown type segment [{:02X} {:02X}] detected. @ {:#08X}", file[ptr], file[ptr + 1], ptr);
                let segment_size = u16::from_be_bytes(<[u8; 2]>::try_from(&file[ptr + 2..ptr + 4]).unwrap());
                print_indent!(1, 2, "segment size: {}", segment_size);
                ptr += 2 + segment_size as usize;
            }
        }
    }
}
