# PopScript

**PopScript** is a programming language inspired by Python and JavaScript, but with its own syntax and features.  
It supports interactive mode, executing `.pop` files, compiling to Python, and building `.exe` files using PyInstaller.

---

## ✨ Features
- **Interpreter** for running `.pop` files.
- **Transpilation to Python** (`.py`) with automatic deletion
- **`.exe` compilation** for Windows (standalone, no interpreter required).
- **PopScript packages** and Python library imports.
- Built-in packages (`time`, etc.).
- Python and JS-like syntax with keywords `var`, `print`, `if/else`, `def`.

---

## 📦 Example Code
```popscript
var e = 2 + 2;
var a = 5;

print(e + a);
print("Hello, World!");
```
## 🚀 Usage
Run a `.pop` file

If you use Source Code use this
```bash
python pop.py myfile.pop --run
```
Else You use from release
```bash
pop myfile.pop -r
```
## How to Build to `.exe`

Again If you use Source Code use this
```bash
python pop.py myfile.pop --executable
```
Else You use from release
```bash
pop myfile.pop -e
```
## Interactive mode
Source Code
```bash
python pop.py
```
Release (.exe)
```bash
pop
```

## 📌 Roadmap
- Static typing (int, string, etc.)
- OOP (classes, objects)
- Parallel execution without GIL
- New package manager Puplease Package (in GitHub)

## ♦ How to Install PopScript
see this video:
There I paste link to video
