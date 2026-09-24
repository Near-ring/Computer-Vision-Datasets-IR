use crate::{Result, invalid};
use geo::{BooleanOps, Coord, LineString, Polygon};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}
impl Point {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
}

/// Half-open pixel-edge box `[x, x + width) × [y, y + height)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}
impl Rect {
    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
    pub fn right(self) -> f32 {
        self.x + self.width
    }
    pub fn bottom(self) -> f32 {
        self.y + self.height
    }
    pub fn area(self) -> f32 {
        self.width * self.height
    }
    pub fn contains(self, p: Point) -> bool {
        p.x >= self.x && p.y >= self.y && p.x < self.right() && p.y < self.bottom()
    }
    pub fn intersection(self, other: Self) -> Option<Self> {
        let x = self.x.max(other.x);
        let y = self.y.max(other.y);
        let width = self.right().min(other.right()) - x;
        let height = self.bottom().min(other.bottom()) - y;
        (width > 0.0 && height > 0.0).then_some(Self::new(x, y, width, height))
    }
    pub fn corners(self) -> Vec<Point> {
        vec![
            Point::new(self.x, self.y),
            Point::new(self.right(), self.y),
            Point::new(self.right(), self.bottom()),
            Point::new(self.x, self.bottom()),
        ]
    }
    pub fn validate(self) -> Result<()> {
        if ![
            self.x,
            self.y,
            self.width,
            self.height,
            self.right(),
            self.bottom(),
            self.area(),
        ]
        .iter()
        .all(|v| v.is_finite())
            || self.width < 0.0
            || self.height < 0.0
        {
            return Err(invalid("non-finite or negative rectangle"));
        }
        Ok(())
    }
    pub fn from_points(points: &[Point]) -> Self {
        if points.is_empty() {
            return Self::default();
        }
        let (mut x0, mut y0, mut x1, mut y1) = (
            f32::INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        );
        for p in points {
            x0 = x0.min(p.x);
            y0 = y0.min(p.y);
            x1 = x1.max(p.x);
            y1 = y1.max(p.y);
        }
        Self::new(x0, y0, x1 - x0, y1 - y0)
    }
}

/// Polygon parts are filled exteriors, not holes. Use a binary mask for holes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Shape {
    Rect(Rect),
    Circle { center: Point, radius: f32 },
    Polygons(Vec<Vec<Point>>),
    Point(Point),
}
impl Shape {
    pub fn bounds(&self) -> Rect {
        match self {
            Self::Rect(r) => *r,
            Self::Circle { center, radius } => Rect::new(
                center.x - radius,
                center.y - radius,
                2.0 * radius,
                2.0 * radius,
            ),
            Self::Polygons(parts) => {
                Rect::from_points(&parts.iter().flatten().copied().collect::<Vec<_>>())
            }
            Self::Point(p) => Rect::new(p.x, p.y, 0.0, 0.0),
        }
    }
    pub fn area(&self) -> f32 {
        match self {
            Self::Rect(r) => r.area(),
            Self::Circle { radius, .. } => std::f32::consts::PI * radius * radius,
            Self::Polygons(p) => p.iter().map(|p| polygon_area(p)).sum(),
            Self::Point(_) => 0.0,
        }
    }
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Rect(r) => r.validate()?,
            Self::Circle { center, radius } => {
                if !center.is_finite() || !radius.is_finite() || *radius <= 0.0 {
                    return Err(invalid("invalid circle"));
                }
            }
            Self::Polygons(parts) => {
                if parts.is_empty()
                    || parts.iter().any(|p| {
                        p.len() < 3 || p.iter().any(|p| !p.is_finite()) || polygon_area(p) <= 0.0
                    })
                {
                    return Err(invalid(
                        "polygon must have at least three finite vertices and positive area",
                    ));
                }
            }
            Self::Point(p) => {
                if !p.is_finite() {
                    return Err(invalid("invalid point"));
                }
            }
        }
        self.bounds().validate()?;
        if !self.area().is_finite() {
            return Err(invalid("geometry area overflow"));
        }
        Ok(())
    }
    /// Tessellates circles; rectangles become four vertices. Points have no polygon.
    pub fn polygons(&self, circle_vertices: usize) -> Vec<Vec<Point>> {
        match self {
            Self::Rect(r) => vec![r.corners()],
            Self::Polygons(p) => p.clone(),
            Self::Point(_) => vec![],
            Self::Circle { center, radius } => vec![
                (0..circle_vertices.max(3))
                    .map(|i| {
                        let a = i as f32 * std::f32::consts::TAU / circle_vertices.max(3) as f32;
                        Point::new(center.x + radius * a.cos(), center.y + radius * a.sin())
                    })
                    .collect(),
            ],
        }
    }
    /// Clips concave polygons using polygon intersection, retaining disconnected parts.
    pub fn clip(&self, bounds: Rect) -> Option<Self> {
        match self {
            Self::Rect(r) => r.intersection(bounds).map(Self::Rect),
            Self::Point(p) => (p.x >= bounds.x
                && p.y >= bounds.y
                && p.x <= bounds.right()
                && p.y <= bounds.bottom())
            .then_some(self.clone()),
            _ => {
                let b = self.bounds();
                if b.x >= bounds.x
                    && b.y >= bounds.y
                    && b.right() <= bounds.right()
                    && b.bottom() <= bounds.bottom()
                {
                    return Some(self.clone());
                }
                b.intersection(bounds)?;
                let clip = geo_polygon(&bounds.corners());
                let parts: Vec<_> = self
                    .polygons(64)
                    .iter()
                    .flat_map(|p| geo_polygon(p).intersection(&clip).0)
                    .filter_map(|p| {
                        let mut v: Vec<_> = p
                            .exterior()
                            .0
                            .iter()
                            .map(|c| Point::new(c.x as f32, c.y as f32))
                            .collect();
                        if v.first() == v.last() {
                            v.pop();
                        }
                        (v.len() >= 3 && polygon_area(&v) > 1e-6).then_some(v)
                    })
                    .collect();
                (!parts.is_empty()).then_some(Self::Polygons(parts))
            }
        }
    }
}
pub(crate) fn polygon_area(p: &[Point]) -> f32 {
    p.iter()
        .zip(p.iter().cycle().skip(1))
        .take(p.len())
        .map(|(a, b)| a.x as f64 * b.y as f64 - b.x as f64 * a.y as f64)
        .sum::<f64>()
        .abs() as f32
        * 0.5
}
fn geo_polygon(p: &[Point]) -> Polygon<f64> {
    Polygon::new(
        LineString::new(
            p.iter()
                .map(|p| Coord {
                    x: p.x as f64,
                    y: p.y as f64,
                })
                .collect(),
        ),
        vec![],
    )
}
