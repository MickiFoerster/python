use rand::Rng;

fn main() {
    let mut rng = rand::thread_rng();

    println!(r#""x", "y""#);
    let mut counter = 0;
    loop {
        let rand_sample = rng.gen_range(0.0..1.0);
        if 0.7 <= rand_sample && rand_sample <= 0.8 {
            println!("{}, 1", rand_sample);

            counter += 1;

            if counter == 100 {
                break;
            }
        } else {
            println!("{}, 0", rand_sample);
        }
    }
}
