# PopScript   --- NO UPDATE ---

## --- REBRANDING ---
- New Project `UltraLang`
---

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
```bash
pop myfile.pop -r
```
## How to Build to `.exe`
```bash
pop myfile.pop -e
```
## Interactive mode
```bash
pop
```

## 📌 Roadmap
- Static typing (int, string, etc.)
- OOP (classes, objects)
- Parallel execution without GIL
