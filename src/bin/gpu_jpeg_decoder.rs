use itertools::Itertools;
use zerocopy::IntoBytes;

static CHECK_ARRAY: fn(&[u8], &[u8]) -> bool =
    |right: &[u8], left: &[u8]| right.iter().zip_eq(left).all(|(a, b)| a == b);

macro_rules! assert_bytes {
    ($ptr:expr,$arr:expr,$literal:expr) => {{
        let literal_len: usize = $literal.len();
        let literal_vec = (0..literal_len)
            .step_by(2)
            .map(|i| u8::from_str_radix(&$literal[i..i + 2], 16).unwrap())
            .collect::<Vec<u8>>();
        let literal_slice = literal_vec.as_slice();
        $arr[$ptr..($ptr + literal_len / 2 as usize)]
            .iter()
            .zip_eq(literal_slice)
            .all(|(a, b)| a == b)
    }};
}
fn main() {
    let file = include_bytes!("test.jpg");
    for word in file.chunks(8) {
        for byte in word {
            print!("{:02X} ", byte);
        }
        println!();
        break;
    }
    let mut ptr = 0;
    

    assert_bytes!(0, file, "FFD8");
}
