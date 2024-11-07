use std::ops;
use image::save_buffer;

const RES: Vect2<usize> = Vect2 { x: 1920, y: 1000 };
const GAMMA: f64 = 1.0 / 2.2;
const FILE_NAME: &str = "shape.png";

fn main() {
    let mut screen: Screen<Rgba> = Screen::new(Rgba(1.0, 1.0, 1.0, 1.0));

    let mut omlaut = Shape {
        color: Rgba(0.0, 0.0, 0.0, 1.0),
        size: 37.5,
        origin: Vect2 { x: 28.0, y: 50.0 },
        contours: vec![
            vec![
                Point { online: true, x: 60.0, y: 0.0 }, Point { online: false, x: 0.0, y: 0.0 },
                Point { online: true, x: 0.0, y: 60.0 }, Point { online: false, x: 0.0, y: 120.0 },
                Point { online: true, x: 60.0, y: 120.0 }, Point { online: false, x: 120.0, y: 120.0 },
                Point { online: true, x: 120.0, y: 60.0 }, Point { online: false, x: 120.0, y: 0.0 },
            ],
            vec![
                Point { online: true, x: 60.0, y: 24.0 }, Point { online: false, x: 24.0, y: 24.0 },
                Point { online: true, x: 24.0, y: 60.0 }, Point { online: false, x: 24.0, y: 96.0 },
                Point { online: true, x: 60.0, y: 96.0 }, Point { online: false, x: 96.0, y: 96.0 },
                Point { online: true, x: 96.0, y: 60.0 }, Point { online: false, x: 96.0, y: 24.0 },
            ],
            vec![
                Point { online: true, x: 40.0, y: 130.0 }, Point { online: false, x: 30.0, y: 130.0 },
                Point { online: true, x: 30.0, y: 140.0 }, Point { online: false, x: 30.0, y: 150.0 },
                Point { online: true, x: 40.0, y: 150.0 }, Point { online: false, x: 50.0, y: 150.0 },
                Point { online: true, x: 50.0, y: 140.0 }, Point { online: false, x: 50.0, y: 130.0 },
            ],
            vec![
                Point { online: true, x: 80.0, y: 130.0 }, Point { online: false, x: 70.0, y: 130.0 },
                Point { online: true, x: 70.0, y: 140.0 }, Point { online: false, x: 70.0, y: 150.0 },
                Point { online: true, x: 80.0, y: 150.0 }, Point { online: false, x: 90.0, y: 150.0 },
                Point { online: true, x: 90.0, y: 140.0 }, Point { online: false, x: 90.0, y: 130.0 },
            ],
        ],
    };

    omlaut.rotate(-0.08);
    omlaut.transform_to_view();
    screen.alpha_over(omlaut.rasterize(), omlaut.color);

    let mut heart = Shape {
        color: Rgba(1.0, 0.02, 0.02, 0.8),
        size: 75.0,
        origin: Vect2 { x: 30.0, y: 27.5 },
        contours: vec![vec![
            Point { online: true, x: 240.0, y: 80.0 },
            Point { online: true, x: 160.0, y: 0.0 },
            Point { online: true, x: 80.0, y: 80.0 }, Point { online: false, x: 0.0, y: 160.0 },
            Point { online: true, x: 40.0, y: 200.0 }, Point { online: false, x: 80.0, y: 240.0 },
            Point { online: true, x: 160.0, y: 160.0 }, Point { online: false, x: 240.0, y: 240.0 },
            Point { online: true, x: 280.0, y: 200.0 }, Point { online: false, x: 320.0, y: 160.0 },
        ]],
    };

    heart.transform_to_view();
    screen.alpha_over(heart.rasterize(), heart.color);

    let mut star = Shape {
        color: Rgba(0.0, 1.0, 0.2, 1.0),
        size: 20.0,
        origin: Vect2 { x: 90.3, y: 35.2 },
        contours: vec![vec![
            Point { online: true, x: -10.0, y: -120.0, },
            Point { online: true, x: 0.0, y: 0.0 },
            Point { online: true, x: -120.0, y: -10.0, },
            Point { online: true, x: -120.0, y: 10.0 },
            Point { online: true, x: 0.0, y: 0.0 },
            Point { online: true, x: -10.0, y: 120.0 },
            Point { online: true, x: 10.0, y: 120.0 },
            Point { online: true, x: 0.0, y: 0.0 },
            Point { online: true, x: 120.0, y: 10.0 },
            Point { online: true, x: 120.0, y: -10.0 },
            Point { online: true, x: 0.0, y: 0.0 },
            Point { online: true, x: 10.0, y: -120.0 },
        ]],
    };

    star.rotate(1.01);
    star.transform_to_view();
    screen.alpha_over(star.rasterize(), star.color);

    save_image(&screen);
}

struct Vect2<T> {
    x: T,
    y: T,
}

// A point specifically within a contour. It is `online` if it is not a control point.
struct Point {
    online: bool,
    x: f64,
    y: f64,
}

impl Point {
    fn scale_then_translate(&mut self, scale: f64, offset: &Vect2<f64>) {
        *self = Point {
            online: self.online,
            x: self.x * scale + offset.x,
            y: self.y * scale + offset.y,
        };
    }
}

struct Screen<T>(Vec<Vec<T>>);

impl<T: Copy + Clone> Screen<T> {
    fn new(value: T) -> Self {
        let mut row: Vec<T> = Vec::with_capacity(RES.x);
        let mut screen: Self = Screen(Vec::with_capacity(RES.y));

        for _ in 0..RES.x {
            row.push(value);
        }

        for _ in 0..RES.y {
            screen.0.push(row.clone());
        }

        screen
    }
}

impl Screen<Rgba> {
    fn alpha_over(&mut self, mask: Screen<f64>, color: Rgba) {
        for y in 0..self.0.len() {
            for x in 0..self.0[y].len() {
                match mask.0[y][x] {
                    0.0 => (),
                    1.0 => self.0[y][x] = self.0[y][x] + color,
                    _ => {
                        let mut new_color = color;
                        new_color.3 *= mask.0[y][x];
                        self.0[y][x] = self.0[y][x] + new_color;
                    },
                }
            }
        }
    }
}

#[derive(Copy, Clone)]
struct Rgba(f64, f64, f64, f64);

impl Rgba {
    fn as_srgb(&self) -> [u8; 3] {
        [
            (self.0.powf(GAMMA) * 255.0) as u8,
            (self.1.powf(GAMMA) * 255.0) as u8,
            (self.2.powf(GAMMA) * 255.0) as u8,
        ]
    }
}

impl ops::Add for Rgba {
    type Output = Self;

    fn add(self, top: Self) -> Self::Output {
        assert!(self.3 == 1.0);

        let alpha = top.3;
        let remaining = 1.0 - alpha;

        Rgba(
            self.0 * remaining + top.0 * alpha,
            self.1 * remaining + top.1 * alpha,
            self.2 * remaining + top.2 * alpha,
            1.0,
        )
    }
}

struct Shape {
    origin: Vect2<f64>,
    size: f64,
    color: Rgba,
    contours: Vec<Vec<Point>>,
}

impl Shape {
    fn transform_to_view(&mut self) {
        let scale_factor = self.size * (RES.y as f64) / 24000.0;
        let offset = Vect2 {
            x: self.origin.x * (RES.y as f64) / 100.0,
            y: self.origin.y * (RES.y as f64) / 100.0,
        };

        for contour in self.contours.iter_mut() {
            for point in contour {
                point.scale_then_translate(scale_factor, &offset);
            }
        }
    }

    fn rotate(&mut self, rotation: f64) {
        let cos_rotation = rotation.cos();
        let sin_rotation = rotation.sin();

        for contour in &mut self.contours {
            for point in contour {
                let new_point_x = point.x * cos_rotation - point.y * sin_rotation;
                point.y = point.x * sin_rotation + point.y * cos_rotation;
                point.x = new_point_x;
            }
        }
    }

    fn y_bounds(&self) -> ops::Range<usize> {
        let mut max = 0.0;
        let mut min = RES.y as f64;

        for contour in &self.contours {
            for point in contour {
                if point.y > max {
                    max = point.y;
                } else if point.y < min {
                    min = point.y;
                }
            }
        }

        max = max.clamp(0.0, (RES.y - 1) as f64);
        min = min.clamp(0.0, RES.y as f64);

        (min as usize)..(1 + max as usize)
    }

    fn rasterize(&self) -> Screen<f64> {
        let mut mask: Screen<f64> = Screen::new(0.0);

        for scanline in self.y_bounds().rev() {
            let mut intersections: Vec<f64> = Vec::new();

            for contour in &self.contours {
                for (i, point) in contour.iter().enumerate() {
                    if !point.online {
                        continue;
                    }

                    let y = scanline as f64 + 0.5;
                    let point2 = &contour[(i + 1) % contour.len()];

                    if point2.online {
                        if let Some(n) = intersect_linear(y, point, point2) {
                            intersections.push(n);
                        }
                    } else {
                        let point3 = &contour[(i + 2) % contour.len()];
                        let roots = intersect_quadratic(y, point, point2, point3);

                        for root in roots {
                            if let Some(n) = root {
                                intersections.push(n);
                            }
                        }
                    }
                }
            }

            intersections.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let row = &mut mask.0[RES.y - scanline - 1];

            fill_row(row, &intersections);
        }

        return mask;
    }
}

fn intersect_linear(y: f64, start: &Point, end: &Point) -> Option<f64> {
    if start.y == end.y
        || start.y > y && end.y > y
        || start.y < y && end.y < y
    {
        return None;
    }

    let t = (y - end.y) / (start.y - end.y);

    if (0.0..=1.0).contains(&t) {
        if start.y > end.y {
            if t == 0.0 {
                return None;
            }
        } else if t == 1.0 {
            return None;
        }

        return Some((start.x - end.x) * t + end.x);
    };

    None
}

fn intersect_quadratic(y: f64, start: &Point, control: &Point, end: &Point) -> [Option<f64>; 2] {
    let mut roots: [Option<f64>; 2] = [None, None];

    if start.y == control.y && start.y == end.y
        || start.y > y && control.y >= y && end.y > y
        || start.y < y && control.y <= y && end.y < y
    {
        return roots;
    }

    let a = start.y - 2.0 * control.y + end.y;
    let b = start.y - control.y;
    let c = y - start.y;
    let discriminant = b * b + a * c;

    if discriminant < 0.0 {
        return roots;
    }

    let roots_y: [f64; 2] = [(b - discriminant.sqrt()) / a, (b + discriminant.sqrt()) / a];

    for (i, root) in roots_y.iter().enumerate() {
        if (0.0..=1.0).contains(root) {
            if (start.y < control.y) && (*root == 0.0) || (end.y < control.y) && (*root == 1.0) {
                continue;
            }

            let t_minus = 1.0 - root;

            roots[i] = Some(
                t_minus * (start.x * t_minus + 2.0 * control.x * root) + end.x * root * root,
            );
        }
    }

    roots
}

fn fill_row(row: &mut [f64], intersections: &[f64]) {
    for i in (0..intersections.len()).step_by(2) {
        let slice_start = (intersections[i] as usize + 1).clamp(0, RES.x - 1);
        let pixel1 = (slice_start - 1).clamp(0, RES.x - 1);
        let slice_end = (intersections[i + 1] as usize).clamp(0, RES.x - 1);
        let pixel2 = (slice_end).clamp(0, RES.x - 1);

        if pixel1 == pixel2 {
            row[pixel1] = intersections[i + 1] - intersections[i];
        } else {
            row[pixel1] = (pixel1 + 1) as f64 - intersections[i];
            row[pixel2] = intersections[i + 1] - pixel2 as f64;
        }

        for x_pos in slice_start..slice_end {
            row[x_pos] = 1.0;
        }
    }
}

fn save_image(screen: &Screen<Rgba>) {
    let mut buffer: Vec<u8> = Vec::with_capacity(RES.x * RES.y * 3);
    let mut color: [u8; 3];

    for row in &screen.0 {
        for pixel in row {
            color = pixel.as_srgb();
            buffer.extend_from_slice(&color);
        }
    }

    save_buffer(
        FILE_NAME,
        &buffer,
        RES.x as u32,
        RES.y as u32,
        image::ColorType::Rgb8,
    ).unwrap();

    println!("`{FILE_NAME}` generated")
}
