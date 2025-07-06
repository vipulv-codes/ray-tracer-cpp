# 🎯 C++ Ray Tracer

A CPU-based ray tracer written from scratch in C++ that renders realistic 3D scenes with:
- Ambient, diffuse, and specular lighting
- Shadows
- Basic reflections
- Plane and sphere geometry
- Configurable camera system

---

## 🛠️ Features

- Written in modern, modular C++
- Ray–shape intersections (Sphere, Plane)
- Phong illumination model
- Shadows & recursive reflections
- Custom camera & perspective setup
- Render output in `.ppm` format

---

## 🖼 Image Conversion

After rendering `.ppm` image using `ray_tracer.cpp`, convert the output to `.png` using `ppmtoimg.py`

---

## ✨ Anti-Aliasing (Supersampling)

To reduce jagged edges (aliasing) along object boundaries, implemented **jittered supersampling anti-aliasing** (cons: increased time complexity many folds, thus slow).