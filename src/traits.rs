pub trait Draw {
    fn draw(&self);
}

pub trait Iterate {
    fn iterate(&self, next: &mut Self);
}

pub trait Update {
    fn update(&mut self) {}
}

pub trait Component: Draw + Update {}
impl<T> Component for T where T: Draw + Update {}
