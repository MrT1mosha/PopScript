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
var popit = 2 + 2;
var simpledimple = 5;

print(popit + simpledimple);
print("Hello, World!");

// Using the built-in time package
package<time> has t;
print(t.now());
t.wait(2);
print("Done!");
```
