// Author: Vipul Vaibhaw

#include<bits/stdc++.h>
#include<memory>
using namespace std;
#define endl '\n'

static constexpr double RAY_T_MIN = 1e-4;   // In order to prevent bouncing rays self-intersecting (shadow acne)
static constexpr double RAY_T_MAX = 1e30;   // 'Infinite' distance
static constexpr double PI = 3.14159265358979323846; // pi
static constexpr int MAX_DEPTH = 5; // max recursion depth for reflections

/* Forward Declarations */
struct vec;
struct ray;
struct Color;
struct Light;
struct Material;
struct Hit;
struct Camera;
class Shape;
class Sphere;
class Plane;
class ShapeSet;

/* Vector */
// 3D Vector with basic arithmetic and geometric operations
struct vec {
    double x, y, z; // three coordinates
   
    vec(double x_=0, double y_=0, double z_=0) : x(x_), y(y_), z(z_) {}
    vec(const vec& v) = default;
    vec& operator=(const vec& v) = default;

    // Arithmetic
    vec operator+(const vec& v) const {
        return vec(x+v.x,y+v.y,z+v.z);
    }
    vec operator-(const vec& v) const {
        return vec(x-v.x,y-v.y,z-v.z);
    }
    vec operator*(double s) const {
        return vec(s*x,s*y,s*z);
    }
    vec operator/(double s) const {
        return vec(x / s, y / s, z / s);
    }
    vec operator-() const {
        return vec(-x, -y, -z);
    }

    // Compound assignments
    vec& operator+=(const vec& v) {
        x += v.x; y += v.y; z += v.z;
        return *this;
    }
    vec& operator-=(const vec& v) {
        x -= v.x; y -= v.y; z -= v.z;
        return *this;
    }
    vec& operator*=(double s) {
        x *= s; y *= s; z *= s;
        return *this;
    }
    vec& operator/=(double s) {
        x /= s; y /= s; z /= s;
        return *this;
    }

    // Geometric
    double dot(const vec& v) const {
        return x*v.x + y*v.y + z*v.z;
    }
    vec cross(const vec& v) const {
        return vec(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        );
    }
    double magnitude() const {
        return sqrt(x*x + y*y + z*z);
    }
    vec unit() const {
        double mag = magnitude();
        if (mag == 0) return vec();
        return *this * (1/mag);
    }

    // Display
    friend ostream& operator<<(ostream& os, const vec& v) {
        return os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    }
};

/* Ray */
// Represents a ray from origin in a direction
struct ray {
    vec origin, direction;
    double tMax;

    ray() : origin(), direction(), tMax(RAY_T_MAX) {}
    ray(const vec& O, const vec& D, double tMax_ = RAY_T_MAX) : origin(O), direction(D.unit()), tMax(tMax_) {}

    ray(const ray& r) = default;
    ray& operator=(const ray& r) = default;

    vec calculate(double t) const {
        return origin + direction * t;
    }
};

/* Color */
// Represents RGB color values (float in range [0, 1]) with utilities for gamma, clamp, etc.
struct Color {
    float r, g, b;

    Color() : r(0), g(0), b(0) {}   // default black
    Color(float l) : r(l), g(l), b(l) {}
    Color(float r_, float g_, float b_) : r(r_), g(g_), b(b_) {}

    // Clamp color values to [min, max]
    void clamp(float min = 0.0f, float max = 1.0f) {
        r = std::clamp(r, min, max);
        g = std::clamp(g, min, max);
        b = std::clamp(b, min, max);
    }

    // Apply gamma correction and exposure adjustment for realistic lighting
    void applyGammaCorrection(float exposure =1.0f, float invgamma =1.0f/2.2f) {
        r = pow(r * exposure, invgamma);
        g = pow(g * exposure, invgamma);
        b = pow(b * exposure, invgamma);
    }

    Color& operator=(const Color& c) {
        r = c.r; g = c.g; b = c.b;
        return *this;
    }

    Color& operator+=(const Color& c) {
        r += c.r; g += c.g; b += c.b;
        return *this;
    }

    Color& operator*=(const Color& c) {
        r *= c.r; g *= c.g; b *= c.b;
        return *this;
    }

    Color& operator*=(float f) {
        r *= f; g *= f; b *= f;
        return *this;
    }

    Color& operator/=(float f) {
        r /= f; g /= f; b /= f;
        return *this;
    }

    // Print color as (r, g, b)
    friend ostream& operator<<(ostream& os, const Color& c) {
        return os << "(" << c.r << ", " << c.g << ", " << c.b << ")";
    }
};

// Additive mixing
inline Color operator+(const Color& a, const Color& b) {
    return Color(a.r + b.r, a.g + b.g, a.b + b.b);
}
// Multiplicative mixing (absorptive)
inline Color operator*(const Color& a, const Color& b) {
    return Color(a.r * b.r, a.g * b.g, a.b * b.b);
}

/* Material */
struct Material {
    Color baseColor;    // true diffuse color
    float k_a;          // ambient strength
    float k_d;          // diffuse strength
    float k_s;          // specular strength
    float shininess;    // Phong exponent
    float reflectivity; // [0,1] mirror factor
};

/* Light */
struct Light {
    vec position; // position of point source
    Color color; // Light color
};

// Scene ambient light and point lights globals
Color ambientLight;
vector<Light> lights;   

/* Hit */
struct Hit {
    ray Ray;
    double t;   // to check for closest hit
    const Shape* hitObject;     // pointer to shape hit
    Color color;    // surface color at the hit.

    Hit() : Ray(), t(RAY_T_MAX), hitObject(nullptr), color(1.0f) {}
    Hit(const ray& r) : Ray(r), t(r.tMax), hitObject(nullptr), color(1.0f) {}

    vec position() const {
        return Ray.calculate(t);    // returns the intersection point
    }
};

/* Shape */
// Abstract base for all geometric shapes in the scene
class Shape {
    public:
        Material mat;   // material properties

        virtual ~Shape() {}

        // Ray–shape intersection:
        virtual bool intersect(Hit& hit) const = 0; // tests hit against `hit.Ray`, updates hit.t/object/color on a closer hit
        virtual bool doesIntersect(const ray& r) const = 0; // returns true if ray `r` hits this shape (does not update a full Hit)
        virtual vec getNormal(const vec &point) const = 0; // Surface normal at a given point on the shape
};

// Sphere shape, defined by its centre and radius
class Sphere : public Shape {
public:
    vec centre;
    double radius;

    Sphere(const vec& c, double r, const Material& m) {
        centre=c; radius=r; mat=m;
    }
    bool intersect(Hit& hit) const override {
        ray localRay = hit.Ray;
        vec OC = localRay.origin - centre;

        double a = localRay.direction.dot(localRay.direction);
        double b = 2 * localRay.direction.dot(OC);
        double c = OC.dot(OC) - radius * radius;

        double discr = b*b - 4*a*c;
        if (discr < 0) return false;

        double sqrtD = sqrt(discr);
        double t1 = (-b - sqrtD) / (2*a);
        double t2 = (-b + sqrtD) / (2*a);
        
        // Choose nearest positive t within valid range
        double t = (t1 >= RAY_T_MIN) ? t1 : ((t2 >= RAY_T_MIN) ? t2 : -1);
        if (t > 0 && t < hit.t) {
            hit.t = t;
            hit.hitObject = this;
            hit.color = mat.baseColor;
            return true;
        }
        return false;
    }

    bool doesIntersect(const ray& r) const override {
        vec OC = r.origin - centre;
        double a = r.direction.dot(r.direction);
        double b = 2 * r.direction.dot(OC);
        double c = OC.dot(OC) - radius * radius;
        double discr = b*b - 4*a*c;
        if (discr < 0) return false;

        double sqrtD = sqrt(discr);
        double t1 = (-b - sqrtD) / (2*a);
        double t2 = (-b + sqrtD) / (2*a);
        return (t1 > RAY_T_MIN && t1 < r.tMax) || (t2 > RAY_T_MIN && t2 < r.tMax);
    }

    vec getNormal(const vec &P) const override {
        return (P - centre).unit(); // radial direction
    }

};

// Plane shape, defined by a point and a normal
class Plane : public Shape {
public:
    vec point;    // any point on the plane
    vec normal;   // unit normal vector to plane
    
    Plane(const vec& p, const vec& n, const Material& m) {
        point = p; normal = n.unit(); mat = m;
    }

    bool intersect(Hit& hit) const override {
        // Ray-plane intersection: t = ((P0 - O)·N)/(D·N)
        double denom = hit.Ray.direction.dot(normal);
        if (fabs(denom) < 1e-6) return false; // ray parallel to plane

        double t = (point - hit.Ray.origin).dot(normal)/denom;
        if (t > RAY_T_MIN && t < hit.t) {
            hit.t = t;
            hit.hitObject = this;
            hit.color = mat.baseColor;
            return true;
        }
        return false;
    }

    bool doesIntersect(const ray& r) const override {
        double denom = r.direction.dot(normal);
        if (fabs(denom) < 1e-6) return false;
        double t = (point - r.origin).dot(normal) / denom;
        return (t > RAY_T_MIN && t < r.tMax);
    }

    vec getNormal(const vec &P) const override {
        return normal;  // same everywhere
    }
};

/* ShapeSet */
// Holds multiple shapes in the set.
class ShapeSet : public Shape {
    vector<unique_ptr<Shape>> shapes;   // Collection of shapes in the set
public:
    // Adds a new shape to the set
    void add(unique_ptr<Shape> shape) {
        shapes.push_back(move(shape));
    }

    // Finds the closest intersection among all shapes in the set
    bool intersect(Hit& hit) const override {
        bool hitAnything = false;
        for (const auto& shape : shapes) {
            if (shape->intersect(hit)) {
                hitAnything = true;
            }
        }
        return hitAnything;
    }

    // Checks if any shape is hit by the ray
    bool doesIntersect(const ray& r) const override {
        for (const auto& shape : shapes) {
            if (shape->doesIntersect(r)) return true;
        }
        return false;
    }

    vec getNormal(const vec & _) const override {
        // This should never be called on the container itself,
        // but provided an implementation to avoid abstractness.
        return vec(0, 0, 0);
    }
};

/* Camera */
// Pinhole camera model: generates rays for a given screen coordinate (u, v)
struct Camera {
    vec origin;     // Position of camera
    vec forward, right, up;     // Camera basis vectors
    double w, h;    // Half-width and half-height of the viewport

    Camera(const vec& origin, const vec& target, const vec& upguide, double fov, double aspect_ratio)
        : origin(origin)
    {
        forward = (target - origin).unit();     // Direction camera is looking
        right = forward.cross(upguide).unit();  // Camera's right(perpendicular to forward & upguide)
        up = right.cross(forward);      // Camera's up
        h = tan(fov);       // half-height of the viewport
        w = h * aspect_ratio; // half-width of the viewport
    }

    // Generates a ray passing through (u, v) on the image plane
    ray makeRay(double u, double v) const {
        vec dir = forward + right * (u * w) + up * (v * h);
        return ray(origin, dir);
    }
};

/* Trace */
// Traces a ray into the scene and computes the final color by shading and reflection
Color trace(const ray &r, const ShapeSet &scene, int depth) {
    if(depth <= 0) return Color(0.0f);
    Hit hit(r);
    if(!scene.intersect(hit)) return ambientLight;  // if no object hit, return ambient

    // Get shading information
    vec P = hit.position();                      // point of intersection
    vec N = hit.hitObject->getNormal(P);         // surface normal at hit point
    vec V = -r.direction;                        // view direction
    auto &M = hit.hitObject->mat;                // material of the hit object

    // ambient contribution
    Color result = M.baseColor * M.k_a * ambientLight;

    // For each light
    for(auto &L : lights) {
        vec Ld = (L.position - P).unit();
        // Shadow check
        if(scene.doesIntersect(ray(P + Ld * RAY_T_MIN, Ld))) continue;

        // Diffuse term: I_L * k_d * (N·L)
        float diff = max(0.0, N.dot(Ld));
        result += (M.baseColor * M.k_d * diff) * L.color;

        // Specular term: I_L * k_s * (R·V)^shininess
        vec R = (N*(2*N.dot(Ld)) - Ld).unit();
        float spec = pow(max(0.0, R.dot(V)), M.shininess);
        result += (Color(1,1,1) * M.k_s * spec) * L.color;
    }

    // Reflection
    if(M.reflectivity > 0 && depth > 0) {
        vec I = r.direction;
        vec Rdir = I - N * 2 * I.dot(N);
        Color refl = trace(ray(P + Rdir * RAY_T_MIN, Rdir), scene, depth-1);
        result += refl * M.reflectivity;
    }

    return result;
}

int main() {
    
    int width = 2000, height = 2000;

    // Scene setup
    ShapeSet scene;

    // Camera setup
    vec cam_pos(0,0,-1);      // Camera position
    vec targ(0,0,1);          // Target reference
    vec world_up(0,1,0);      // Upward direction for camera orientation
    Camera camera(cam_pos, targ, world_up, PI/4, double(width)/height);

    // Lighting setup
    ambientLight = Color(0.05f);     // Global illumination
    lights.push_back({vec(1,1,-1), Color(1.0f)});
    lights.push_back({vec(-1,1,-0.5), Color(0.6f,0.8f,1.0f)});

    // Material setup: baseColor, k_a, k_d, k_s, shininess, reflectivity
    Material redMat   = { Color(1,0.2f,0.2f), 0.2f, 0.8f, 0.5f, 32.0f, 0.0f };
    Material greenMat = { Color(0.2f,1,0.2f), 0.2f, 0.8f, 0.5f, 32.0f, 0.0f };
    Material blueMat  = { Color(0.2f,0.2f,1), 0.2f, 0.8f, 0.5f, 32.0f, 0.0f };
    Material floorMat = { Color(0.5f),        0.2f, 0.8f, 0.5f, 32.0f, 0.5f };

    // Objects with specific materials
    scene.add(make_unique<Sphere>(vec(0,0,1), 0.3, redMat));
    scene.add(make_unique<Sphere>(vec(0.6,0,1.2), 0.3, greenMat));
    scene.add(make_unique<Sphere>(vec(-0.6,0,1.2), 0.3, blueMat));
    scene.add(make_unique<Plane>(vec(0,-0.5,0), vec(0,1,0), floorMat));

    // Rendering
    ofstream img("render.ppm");
    img << "P3\n" << width << " " << height << "\n255\n";

    // Anti-Aliasing (SuperSampling)
    mt19937 rng(random_device{}());
    uniform_real_distribution<double> uniform(0.0,1.0); // uniform [0,1)

    bool AA = true;
    int ray_per_pixel = AA ? 8 : 1;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Color col;
            for (int i = 0; i < ray_per_pixel; ++i) {
                double dx = 0.5, dy = 0.5;
                if (AA) {
                    // jittering
                    dx = uniform(rng);
                    dy = uniform(rng);
                }

                // double u = ((x+0.5) - width/2.0) / (width/2.0);   
                // double v = (height/2.0 - (y+0.5)) / (height/2.0);

                double u = ((x+dx) - width/2.0) / (width/2.0);   
                double v = (height/2.0 - (y+dy)) / (height/2.0);

                ray r = camera.makeRay(u,v);
                col += trace(r, scene, MAX_DEPTH);
            }
            col /= ray_per_pixel;
            col.clamp();
            col.applyGammaCorrection();

            int R = int(col.r * 255);
            int G = int(col.g * 255);
            int B = int(col.b * 255);
            img << R << " " << G << " " << B << " ";
        }
        img << "\n";
    }
    img.close();
    cout << "Rendered image saved as render.ppm!" << endl;

    return 0;
}