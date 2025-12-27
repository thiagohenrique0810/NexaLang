# Create test file for methods
code = '''struct Point {
    x: i32,
    y: i32
}

impl Point {
    // Static method
    fn new(x: i32, y: i32) -> Point {
        return Point(x, y);
    }
    
    // Instance method with &self
    fn distance(&self) -> i32 {
        return self.x * self.x + self.y * self.y;
    }
    
    // Instance method with &mut self
    fn move(&mut self, dx: i32, dy: i32) {
        self.x = self.x + dx;
        self.y = self.y + dy;
    }
}

fn main() -> i32 {
    let p: Point = Point::new(3, 4);
    print(p.distance());  // Should print 25
    
    let mut q: Point = Point::new(0, 0);
    q.move(10, 20);
    print(q.x);  // Should print 10
    print(q.y);  // Should print 20
    
    return 0;
}
'''

with open('examples/methods_test.nxl', 'w') as f:
    f.write(code)

print("Created examples/methods_test.nxl")
