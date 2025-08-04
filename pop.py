import sys
import argparse
import os
import shutil
import subprocess
import importlib
import datetime
import time as py_time
from PopWindow import PopWindow

class Token:
    def __init__(self, type, value=None):
        self.type = type
        self.value = value

    def __repr__(self):
        return f"Token({self.type}, {self.value})"

# Define the types of tokens we will recognize
TOKENS = {
    'NUMBER': r'\d+',
    'STRING': r'"[^"]*"',
    'VAR_DECL': 'var',
    'IF': 'if',
    'ELSE': 'else',
    'FOR': 'for',
    'IN': 'in',
    'DEF': 'def',
    'IDENTIFIER': r'[a-zA-Z_][a-zA-Z0-9_]*',
    'EQUAL': '=',
    'PLUS': r'\+',
    'MINUS': r'-',
    'MULTIPLY': r'\*',
    'DIVIDE': r'/',
    'DOT': r'\.',
    'LPAREN': r'\((?!<)',
    'RPAREN': r'\)',
    'LBRACE': r'{',
    'RBRACE': r'}',
    'SEMICOLON': r';',
    'COMMA': r',',
    'EQ_EQ': r'==',
    'NEQ': r'!=',
    'LT': r'<',
    'GT': r'>',
    'LTE': r'<=',
    'GTE': r'>=',
    'PRINT': 'print',
    'PACKAGE': 'package',
    'HAS': 'has',
}

# --- Mock Library Files (for demonstration) ---
# For demonstration purposes, we will simulate the file system by placing the
# content of 'core/numbers.pop' into a string. The interpreter will now
# check for a file, and if it doesn't exist, it will use this mock content.
# In a real-world app, you would remove this dictionary.
POPSCRIPT_NATIVE_MOCK_FILES = {
    "core/numbers.pop": "def is_even(num) { return num % 2 == 0; } def is_odd(num) { return num % 2 != 0; }"
}

# --- Built-in PopScript Packages ---
# These packages are available directly in PopScript and map to Python functions.
BUILT_IN_PACKAGES = {
    "time": {
        "now": lambda: datetime.datetime.now().strftime("%H:%M:%S"),
        "date": lambda: datetime.datetime.now().strftime("%Y-%m-%d"),
        "wait": lambda seconds: py_time.sleep(seconds) # This is the function that handles time.wait
    }
}

class Lexer:
    """
    The Lexer converts the PopScript source code into a list of tokens.
    It removes whitespace and recognizes keywords, identifiers, and operators.
    """
    def __init__(self, text):
        self.text = text
        self.position = 0
        if text:
            self.current_char = self.text[self.position]
        else:
            self.current_char = None

    def advance(self):
        """Move to the next character in the source code."""
        self.position += 1
        if self.position < len(self.text):
            self.current_char = self.text[self.position]
        else:
            self.current_char = None

    def peek(self):
        """Look at the next character without advancing."""
        peek_pos = self.position + 1
        if peek_pos < len(self.text):
            return self.text[peek_pos]
        return None

    def skip_whitespace(self):
        """Skip over any whitespace characters."""
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def get_number(self):
        """Extract a number from the source code."""
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        return Token('NUMBER', int(result))

    def get_string(self):
        """Extract a string from the source code."""
        result = ''
        self.advance()  # Skip the opening quote
        while self.current_char is not None and self.current_char != '"':
            result += self.current_char
            self.advance()
        self.advance()  # Skip the closing quote
        return Token('STRING', result)

    def get_identifier_or_keyword(self):
        """Extract an identifier or a keyword."""
        result = ''
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
            result += self.current_char
            self.advance()
        
        # Check if the identifier is a reserved keyword
        if result == 'var': return Token('VAR_DECL')
        if result == 'if': return Token('IF')
        if result == 'else': return Token('ELSE')
        if result == 'for': return Token('FOR')
        if result == 'in': return Token('IN')
        if result == 'def': return Token('DEF')
        if result == 'print': return Token('PRINT')
        if result == 'package': return Token('PACKAGE')
        if result == 'has': return Token('HAS')
        
        return Token('IDENTIFIER', result)

    def get_next_token(self):
        """The main method to produce the next token."""
        while self.current_char is not None:
            self.skip_whitespace()
            if self.current_char is None:
                break
            
            if self.current_char == '/':
                # Check for a single-line comment
                if self.peek() == '/':
                    while self.current_char is not None and self.current_char != '\n':
                        self.advance()
                    continue
            
            if self.current_char.isdigit(): return self.get_number()
            if self.current_char == '"': return self.get_string()
            if self.current_char.isalpha() or self.current_char == '_': return self.get_identifier_or_keyword()
            if self.current_char == '=':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token('EQ_EQ', '==')
                return Token('EQUAL', '=')
            if self.current_char == '+':
                self.advance()
                return Token('PLUS', '+')
            if self.current_char == '-':
                self.advance()
                return Token('MINUS', '-')
            if self.current_char == '*':
                self.advance()
                return Token('MULTIPLY', '*')
            if self.current_char == '/':
                self.advance()
                return Token('DIVIDE', '/')
            if self.current_char == '(':
                self.advance()
                return Token('LPAREN', '(')
            if self.current_char == ')':
                self.advance()
                return Token('RPAREN', ')')
            if self.current_char == '{':
                self.advance()
                return Token('LBRACE', '{')
            if self.current_char == '}':
                self.advance()
                return Token('RBRACE', '}')
            if self.current_char == ';':
                self.advance()
                return Token('SEMICOLON', ';')
            if self.current_char == ',':
                self.advance()
                return Token('COMMA', ',')
            if self.current_char == '.':
                self.advance()
                return Token('DOT', '.')
            if self.current_char == '<':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token('LTE', '<=')
                return Token('LT', '<')
            if self.current_char == '>':
                self.advance()
                if self.current_char == '=':
                    self.advance()
                    return Token('GTE', '>=')
                return Token('GT', '>')
            
            raise Exception(f"Invalid character: {self.current_char} at position {self.position}")
        
        return Token('EOF')

    def tokenize(self):
        """Create a full list of tokens from the source text."""
        tokens = []
        while True:
            token = self.get_next_token()
            tokens.append(token)
            if token.type == 'EOF':
                break
        return tokens

# --- Abstract Syntax Tree (AST) ---
# The parser will build a tree-like structure from the tokens.
# Each class represents a node in this tree.
class AST:
    pass

class Program(AST):
    def __init__(self, statements):
        self.statements = statements

class VarDecl(AST):
    def __init__(self, name, value):
        self.name = name
        self.value = value

class Assign(AST):
    def __init__(self, name, value):
        self.name = name
        self.value = value

class BinOp(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

class Num(AST):
    def __init__(self, value):
        self.value = value

class String(AST):
    def __init__(self, value):
        self.value = value

class Var(AST):
    def __init__(self, name):
        self.name = name

class Block(AST):
    def __init__(self, statements):
        self.statements = statements

class IfStatement(AST):
    def __init__(self, condition, then_block, else_block=None):
        self.condition = condition
        self.then_block = then_block
        self.else_block = else_block

class PrintStatement(AST):
    def __init__(self, expression):
        self.expression = expression

class PackageStatement(AST):
    def __init__(self, package_name, alias=None):
        self.package_name = package_name
        self.alias = alias
        
class MemberAccess(AST):
    def __init__(self, obj, member_name):
        self.obj = obj
        self.member_name = member_name

class FunctionCall(AST):
    def __init__(self, callable_expr, arguments):
        self.callable_expr = callable_expr
        self.arguments = arguments

# --- PopScript Parser ---
# The parser builds the AST from the tokens produced by the lexer.
class Parser:
    """
    The Parser takes a list of tokens and builds an Abstract Syntax Tree (AST).
    """
    def __init__(self, tokens):
        self.tokens = tokens
        self.token_index = 0
        if tokens:
            self.current_token = self.tokens[self.token_index]
        else:
            self.current_token = Token('EOF')

    def advance(self):
        """Move to the next token."""
        self.token_index += 1
        if self.token_index < len(self.tokens):
            self.current_token = self.tokens[self.token_index]
        else:
            self.current_token = Token('EOF')

    def eat(self, token_type):
        """Consume the current token if it matches the expected type."""
        if self.current_token.type == token_type:
            self.advance()
        else:
            raise Exception(f"Expected token {token_type} but got {self.current_token.type}")

    def atom(self):
        """Handle numbers, strings, and parenthesized expressions."""
        token = self.current_token
        if token.type == 'NUMBER':
            self.eat('NUMBER')
            return Num(token.value)
        elif token.type == 'STRING':
            self.eat('STRING')
            return String(token.value)
        elif token.type == 'IDENTIFIER':
            self.eat('IDENTIFIER')
            return Var(token.value)
        elif token.type == 'LPAREN':
            self.eat('LPAREN')
            node = self.expr()
            self.eat('RPAREN')
            return node
        
        raise Exception(f"Unexpected token in atom: {token.type}")

    def factor(self):
        """Handle member access and function calls."""
        node = self.atom()
        while self.current_token.type in ('DOT', 'LPAREN'):
            if self.current_token.type == 'DOT':
                self.eat('DOT')
                member_name_token = self.current_token
                self.eat('IDENTIFIER')
                node = MemberAccess(obj=node, member_name=member_name_token.value)
            elif self.current_token.type == 'LPAREN':
                self.eat('LPAREN')
                arguments = self.argument_list()
                self.eat('RPAREN')
                node = FunctionCall(callable_expr=node, arguments=arguments)
        return node
    
    def argument_list(self):
        """Parse a list of arguments for a function call."""
        arguments = []
        if self.current_token.type != 'RPAREN':
            arguments.append(self.expr())
            while self.current_token.type == 'COMMA':
                self.eat('COMMA')
                arguments.append(self.expr())
        return arguments

    def term(self):
        """Handle multiplication and division."""
        node = self.factor()
        while self.current_token.type in ('MULTIPLY', 'DIVIDE'):
            token = self.current_token
            if token.type == 'MULTIPLY':
                self.eat('MULTIPLY')
            elif token.type == 'DIVIDE':
                self.eat('DIVIDE')
            node = BinOp(left=node, op=token, right=self.factor())
        return node

    def expr(self):
        """Handle addition and subtraction."""
        node = self.term()
        while self.current_token.type in ('PLUS', 'MINUS'):
            token = self.current_token
            if token.type == 'PLUS':
                self.eat('PLUS')
            elif token.type == 'MINUS':
                self.eat('MINUS')
            node = BinOp(left=node, op=token, right=self.term())
        return node
    
    def condition(self):
        """Parse a conditional expression (e.g., x > 5)."""
        left = self.expr()
        op = self.current_token
        if op.type in ('EQ_EQ', 'NEQ', 'LT', 'GT', 'LTE', 'GTE'):
            self.advance()
            right = self.expr()
            return BinOp(left=left, op=op, right=right)
        
        # If no comparison operator, treat the expression itself as the condition
        return left

    def package_statement(self):
        """Parse a package import statement, e.g., 'package<tkinter> has tk;'"""
        self.eat('PACKAGE')
        self.eat('LT') # Consume the '<'
        package_name_token = self.current_token
        if package_name_token.type not in ('IDENTIFIER', 'STRING'):
            raise Exception(f"Expected identifier or string for package name, got {package_name_token.type}")
        package_name = package_name_token.value
        self.advance()
        self.eat('GT') # Consume the '>'

        alias = None
        if self.current_token.type == 'HAS':
            self.eat('HAS')
            alias_token = self.current_token
            self.eat('IDENTIFIER')
            alias = alias_token.value
        
        self.eat('SEMICOLON')
        return PackageStatement(package_name, alias)

    def statement(self):
        """Parse a single statement."""
        if self.current_token.type == 'PACKAGE':
            return self.package_statement()
        elif self.current_token.type == 'VAR_DECL':
            self.eat('VAR_DECL')
            var_name = self.current_token.value
            self.eat('IDENTIFIER')
            self.eat('EQUAL')
            value_node = self.expr()
            self.eat('SEMICOLON')
            return VarDecl(var_name, value_node)
        elif self.current_token.type == 'IDENTIFIER':
            # Check for assignment
            peek_token_index = self.token_index + 1
            if peek_token_index < len(self.tokens) and self.tokens[peek_token_index].type == 'EQUAL':
                var_name = self.current_token.value
                self.eat('IDENTIFIER')
                self.eat('EQUAL')
                value_node = self.expr()
                self.eat('SEMICOLON')
                return Assign(var_name, value_node)
            else:
                # Handle standalone expressions like function calls
                expr_node = self.expr()
                self.eat('SEMICOLON')
                return expr_node
        elif self.current_token.type == 'IF':
            self.eat('IF')
            self.eat('LPAREN')
            condition_node = self.condition()
            self.eat('RPAREN')
            then_block = self.block()
            else_block = None
            if self.current_token.type == 'ELSE':
                self.eat('ELSE')
                else_block = self.block()
            return IfStatement(condition_node, then_block, else_block)
        elif self.current_token.type == 'PRINT':
            self.eat('PRINT')
            self.eat('LPAREN')
            expr_node = self.expr()
            self.eat('RPAREN')
            self.eat('SEMICOLON')
            return PrintStatement(expr_node)
        else:
            # Handle empty statements (e.g., just a semicolon)
            if self.current_token.type == 'SEMICOLON':
                self.eat('SEMICOLON')
                return None # Return None for an empty statement
            raise Exception(f"Unexpected token in statement: {self.current_token.type}")

    def block(self):
        """Parse a block of code enclosed in braces."""
        self.eat('LBRACE')
        statements = []
        while self.current_token.type != 'RBRACE' and self.current_token.type != 'EOF':
            statement = self.statement()
            if statement: # Don't add None to the list of statements
                statements.append(statement)
        self.eat('RBRACE')
        return Block(statements)

    def parse(self):
        """The main method to build the entire AST."""
        statements = []
        while self.current_token.type != 'EOF':
            statement = self.statement()
            if statement: # Don't add None to the list of statements
                statements.append(statement)
        return Program(statements)

# --- PopScript Interpreter and Translator ---
class Interpreter:
    """
    The Interpreter executes the AST generated by the parser.
    It uses a symbol table to store variables.
    """
    def __init__(self, tree, symbol_table=None):
        self.tree = tree
        self.symbol_table = symbol_table if symbol_table is not None else {}

    def visit(self, node):
        """Dispatch a method based on the node type."""
        method_name = f'visit_{type(node).__name__}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception(f"No visit_{type(node).__name__} method")

    def visit_Program(self, node):
        for statement in node.statements:
            self.visit(statement)

    def visit_PackageStatement(self, node):
        package_name = node.package_name
        alias = node.alias

        # Handle Built-in PopScript Packages
        if package_name in BUILT_IN_PACKAGES:
            imported_lib = BUILT_IN_PACKAGES[package_name]
            # If an alias is provided, create a nested dictionary in the symbol table
            if alias:
                self.symbol_table[alias] = imported_lib
            # Otherwise, add all functions to the global scope
            else:
                self.symbol_table.update(imported_lib)
            print(f"PopScript built-in package '{package_name}' loaded successfully.")
            return

        # Handle PopScript Native Files - now reads from the file system
        if package_name.endswith('.pop'):
            # First, check if the file exists on the disk.
            pop_file_path = package_name
            if os.path.exists(pop_file_path):
                try:
                    with open(pop_file_path, 'r') as f:
                        pop_file_content = f.read()
                except Exception as e:
                    raise Exception(f"Failed to read PopScript file '{pop_file_path}': {e}")
            elif package_name in POPSCRIPT_NATIVE_MOCK_FILES:
                # Fallback to the mock file for demonstration
                pop_file_content = POPSCRIPT_NATIVE_MOCK_FILES[package_name]
            else:
                raise Exception(f"PopScript file '{pop_file_path}' not found.")
            
            # Create a new symbol table for the imported package
            package_symbol_table = {}
            
            # Re-use the lexer and parser to process the file's content
            lexer = Lexer(pop_file_content)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
            package_ast = parser.parse()
            
            # Interpret the package in its own scope
            package_interpreter = Interpreter(package_ast, package_symbol_table)
            package_interpreter.interpret()
            
            # Expose the package's symbol table to the main script
            base_name = os.path.splitext(os.path.basename(package_name))[0]
            alias_name = alias if alias else base_name
            self.symbol_table[alias_name] = package_symbol_table
            print(f"PopScript file '{package_name}' imported as '{alias_name}'.")
            return
            
        # --- This section supports all Python libraries ---
        # The interpreter attempts to import the module. If it succeeds, it's added to the symbol table.
        # This is how PopScript gains access to any installed Python package.
        try:
            imported_module = importlib.import_module(package_name)
            # If an alias is provided, add the module to the symbol table with the alias
            if alias:
                self.symbol_table[alias] = imported_module
            # Otherwise, add the module directly by its name
            else:
                self.symbol_table[package_name] = imported_module
            print(f"Python package '{package_name}' loaded successfully.")
        except ImportError:
            raise Exception(f"Failed to import Python package: {package_name}")

    def visit_VarDecl(self, node):
        self.symbol_table[node.name] = self.visit(node.value)

    def visit_Assign(self, node):
        if node.name not in self.symbol_table:
            raise Exception(f"Undefined variable: {node.name}")
        self.symbol_table[node.name] = self.visit(node.value)

    def visit_BinOp(self, node):
        left_val = self.visit(node.left)
        right_val = self.visit(node.right)

        if node.op.type == 'PLUS': return left_val + right_val
        if node.op.type == 'MINUS': return left_val - right_val
        if node.op.type == 'MULTIPLY': return left_val * right_val
        if node.op.type == 'DIVIDE': return left_val / right_val
        
        if node.op.type == 'EQ_EQ': return left_val == right_val
        if node.op.type == 'NEQ': return left_val != right_val
        if node.op.type == 'LT': return left_val < right_val
        if node.op.type == 'GT': return left_val > right_val
        if node.op.type == 'LTE': return left_val <= right_val
        if node.op.type == 'GTE': return left_val >= right_val

    def visit_Num(self, node):
        return node.value

    def visit_String(self, node):
        return node.value

    def visit_Var(self, node):
        value = self.symbol_table.get(node.name)
        if value is None:
            raise Exception(f"Undefined variable: {node.name}")
        return value

    def visit_MemberAccess(self, node):
        obj = self.visit(node.obj)
        member_name = node.member_name
        
        # This handles Python modules and our mock packages (dictionaries)
        try:
            return getattr(obj, member_name)
        except AttributeError:
            try:
                return obj[member_name]
            except (TypeError, KeyError):
                raise Exception(f"Object '{obj}' has no member '{member_name}'")

    def visit_FunctionCall(self, node):
        callable_obj = self.visit(node.callable_expr)
        arguments = [self.visit(arg) for arg in node.arguments]
        
        if not callable(callable_obj):
            raise Exception(f"Expression '{callable_obj}' is not a callable function.")
        
        return callable_obj(*arguments)

    def visit_Block(self, node):
        for statement in node.statements:
            self.visit(statement)

    def visit_IfStatement(self, node):
        if self.visit(node.condition):
            self.visit(node.then_block)
        elif node.else_block:
            self.visit(node.else_block)

    def visit_PrintStatement(self, node):
        print(self.visit(node.expression))

    def interpret(self):
        """Start the interpretation process from the root of the AST."""
        return self.visit(self.tree)

class Translator:
    """
    The Translator converts the PopScript AST into Python code strings.
    """
    def __init__(self):
        self.code = []
        self.indent_level = 0
    
    def _indent(self):
        return "    " * self.indent_level
        
    def translate(self, node):
        method_name = f'translate_{type(node).__name__}'
        translator = getattr(self, method_name, self.generic_translate)
        return translator(node)

    def generic_translate(self, node):
        raise Exception(f"No translate_{type(node).__name__} method")
        
    def translate_Program(self, node):
        # We need to add the necessary imports at the top
        self.code.append("import time as py_time\n") # Import for time.wait
        self.code.append("import os\n")
        self.code.append("import shutil\n")
        
        for statement in node.statements:
            self.translate(statement)
        return "\n".join(self.code)

    def translate_PackageStatement(self, node):
        package_name = node.package_name
        alias = node.alias
        
        if package_name.endswith('.pop'):
            self.code.append(f"{self._indent()}# PopScript native package import '{package_name}' skipped for translation.")
        else:
            if alias:
                self.code.append(f"{self._indent()}import {package_name} as {alias}")
            else:
                self.code.append(f"{self._indent()}import {package_name}")

    def translate_VarDecl(self, node):
        self.code.append(f"{self._indent()}{node.name} = {self.translate(node.value)}")

    def translate_Assign(self, node):
        self.code.append(f"{self._indent()}{node.name} = {self.translate(node.value)}")

    def translate_BinOp(self, node):
        op_map = {
            'PLUS': '+', 'MINUS': '-', 'MULTIPLY': '*', 'DIVIDE': '/',
            'EQ_EQ': '==', 'NEQ': '!=', 'LT': '<', 'GT': '>',
            'LTE': '<=', 'GTE': '>='
        }
        left = self.translate(node.left)
        right = self.translate(node.right)
        op_symbol = op_map.get(node.op.type)
        return f"({left} {op_symbol} {right})"

    def translate_Num(self, node):
        return str(node.value)

    def translate_String(self, node):
        return f'"{node.value}"'

    def translate_Var(self, node):
        return node.name

    def translate_MemberAccess(self, node):
        obj_code = self.translate(node.obj)
        member_name = node.member_name
        return f"{obj_code}.{member_name}"

    def translate_FunctionCall(self, node):
        callable_code = self.translate(node.callable_expr)
        arguments_code = [self.translate(arg) for arg in node.arguments]
        
        # This is a special case to translate the PopScript 'time.wait' function to Python's 'time.sleep'
        if callable_code == "time.wait":
            callable_code = "py_time.sleep"

        return f"{callable_code}({', '.join(arguments_code)})"

    def translate_Block(self, node):
        self.indent_level += 1
        for statement in node.statements:
            self.translate(statement)
        self.indent_level -= 1

    def translate_IfStatement(self, node):
        self.code.append(f"{self._indent()}if {self.translate(node.condition)}:")
        self.translate(node.then_block)
        if node.else_block:
            self.code.append(f"{self._indent()}else:")
            self.translate(node.else_block)
            
    def translate_PrintStatement(self, node):
        self.code.append(f"{self._indent()}print({self.translate(node.expression)})")

def interactive_mode():
    """
    Runs the PopScript interpreter in an interactive console mode.
    """
    print("Welcome to PopScript! (v0.3)")
    print("Type 'exit()' or 'quit()' to quit.")

    # Maintain a single symbol table for the entire session
    symbol_table = {}

    while True:
        try:
            line = input(">>> ")
            if line.strip().lower() in ('exit()', 'quit()'):
                print("Exiting PopScript console.")
                break
            
            lexer = Lexer(line)
            tokens = lexer.tokenize()
            parser_obj = Parser(tokens)
            ast = parser_obj.parse()
            
            interpreter = Interpreter(ast, symbol_table)
            interpreter.interpret()

        except Exception as e:
            print(f"Error: {e}")
        except KeyboardInterrupt:
            print("\nExiting PopScript console.")
            break

def build_executable(popscript_file, ast):
    """
    Translates a PopScript file to Python, then builds a standalone executable.
    It cleans up intermediate files and only deletes the original source on success.
    """
    print(f"\n--- PopScript Executable Build ---")
    
    # Step 1: Translate to Python
    translator = Translator()
    python_code = translator.translate(ast)
    output_filename_py = os.path.splitext(popscript_file)[0] + '.py'
    
    with open(output_filename_py, 'w') as f:
        f.write(python_code)
    
    print(f"Generating Python file: {output_filename_py}")
    
    try:
        # Step 2: Use PyInstaller to create the executable
        print("Compiling to a standalone executable...")
        # PyInstaller output is redirected to the console for better user feedback
        subprocess.run(["pyinstaller", "--onefile", output_filename_py], check=True)
        
        # Determine the name of the executable to check for
        executable_name = os.path.splitext(os.path.basename(popscript_file))[0]
        if os.name == 'nt':  # Windows
            executable_name += '.exe'
        
        # Step 3: Verify the executable was created
        if os.path.exists(os.path.join('dist', executable_name)):
            print("Executable created successfully.")
            
            # Step 4: Clean up intermediate files
            print("Cleaning up intermediate files...")
            os.remove(output_filename_py)
            shutil.rmtree('build', ignore_errors=True)
            spec_file = f'{os.path.splitext(output_filename_py)[0]}.spec'
            if os.path.exists(spec_file):
                os.remove(spec_file)
            
            # Step 5: Delete the original PopScript file
            os.remove(popscript_file)
            
            print("---------------------------------")
            print(f"Standalone executable created. Look for '{executable_name}' in the 'dist' directory.")
            print(f"Original file '{popscript_file}' has been deleted.")
            print("---------------------------------")
        else:
            print("Error: PyInstaller ran but did not produce the expected executable.")
            print("The original PopScript file was not deleted.")
            print("---------------------------------")
            
    except subprocess.CalledProcessError as e:
        print("---------------------------------")
        print(f"Error: PyInstaller failed to create the executable.")
        print("Please check the output above for details.")
        print(f"The original PopScript file '{popscript_file}' was not deleted.")
        print("---------------------------------")
        
    except FileNotFoundError:
        print("---------------------------------")
        print("Error: PyInstaller not found.")
        print("Please install it with `pip install pyinstaller`.")
        print(f"The original PopScript file '{popscript_file}' was not deleted.")
        print("---------------------------------")
    finally:
        # This block ensures the temporary Python file is always removed
        if os.path.exists(output_filename_py):
            os.remove(output_filename_py)


# --- Main execution logic ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PopScript Interpreter and Compiler.")
    parser.add_argument("file", nargs='?', help="The PopScript file to process.")
    
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument("-r", "--run", action="store_true", help="Run the PopScript file.")
    action_group.add_argument("-b", "--build", action="store_true", help="Build the PopScript file into a Python script and delete the original file.")
    action_group.add_argument("-e", "--executable", action="store_true", help="Build a standalone executable from the PopScript file.")

    args = parser.parse_args()

    # If no file is provided and no action is specified, enter interactive mode
    if args.file is None and not args.run and not args.build and not args.executable:
        interactive_mode()
        sys.exit(0)
    
    # If a file is provided, process it
    if args.file:
        popscript_file = args.file
        if not os.path.exists(popscript_file):
            print(f"Error: The file '{popscript_file}' was not found.")
            sys.exit(1)

        try:
            with open(popscript_file, 'r') as file:
                script_content = file.read()
        except Exception as e:
            print(f"An unexpected error occurred while reading the file: {e}")
            sys.exit(1)

        # First, tokenize and parse the code regardless of the action
        lexer = Lexer(script_content)
        tokens = lexer.tokenize()
        parser_obj = Parser(tokens)
        ast = parser_obj.parse()

        if args.build:
            # Build the PopScript into a Python file
            translator = Translator()
            python_code = translator.translate(ast)

            output_filename = os.path.splitext(popscript_file)[0] + '.py'
            with open(output_filename, 'w') as f:
                f.write(python_code)
            
            # Delete the original PopScript file as requested
            os.remove(popscript_file)
            
            print(f"\n--- PopScript Build Complete ---")
            print(f"PopScript translated to Python successfully. Output file: {output_filename}")
            print(f"Original file '{popscript_file}' has been deleted.")
            print("---------------------------------")
        
        elif args.executable:
            # Build a standalone executable using PyInstaller
            build_executable(popscript_file, ast)
        
        else: # Default action is to run
            # Run the PopScript file using the interpreter
            interpreter = Interpreter(ast)
            print("\n--- PopScript Output ---")
            interpreter.interpret()
            print("------------------------")
    else:
        # If an action is specified without a file, it's an invalid command
        print("Error: An action flag (--run, --build, or --executable) requires a file argument.")
        parser.print_help()
        sys.exit(1)