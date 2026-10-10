"""Small expression grammar; slash fractions, single-character variables, scripts."""
from dataclasses import dataclass
from .library import LABELS
from ..math_render import SYMBOLS

class SyntaxError(ValueError): pass
class UnsupportedSyntax(ValueError): pass

@dataclass
class Node:
    kind: str
    value: str = ''
    children: tuple = ()

class Parser:
    def __init__(self, text):
        if not text.strip(): raise SyntaxError('Empty equation')
        if len(text)>1000: raise SyntaxError('Equation too long (1000 characters maximum)')
        if '\\' in text: raise UnsupportedSyntax('LaTeX uses the original font renderer')
        translations = {'²':'^{2}','³':'^{3}','₀':'_{0}','₁':'_{1}','₂':'_{2}'}
        for char,replacement in translations.items(): text=text.replace(char,replacement)
        self.tokens = [c for c in text if not c.isspace()]
        allowed = set(LABELS) | set('/^_{}*')
        if any(c not in allowed for c in self.tokens): raise UnsupportedSyntax('Character outside restricted syntax')
        self.i=0;self.depth=0

    def peek(self): return self.tokens[self.i] if self.i<len(self.tokens) else ''
    def take(self):
        value=self.peek()
        if not value: raise SyntaxError('Unexpected end of expression')
        self.i+=1;return value

    def parse(self):
        node=self.sequence()
        if self.peek(): raise SyntaxError('Unexpected token '+self.peek())
        return node

    def sequence(self):
        children=[self.product()]
        while self.peek() and self.peek() in '+−-=⇒→≤≥≠±':
            op=self.take(); children.extend((Node('glyph','−' if op=='-' else op),self.product()))
        return children[0] if len(children)==1 else Node('row',children=tuple(children))

    def product(self):
        children=[self.factor()]
        while self.peek() and self.peek() not in '+−-=⇒→≤≥≠±)]}':
            if self.peek()=='/':
                self.take()
                numerator=children[0] if len(children)==1 else Node('row',children=tuple(children))
                children=[Node('fraction',children=(numerator,self.factor()))]
            elif self.peek() in '*×÷':
                op=self.take();children.extend((Node('glyph','×' if op=='*' else op),self.factor()))
            else: children.append(self.factor())
        return children[0] if len(children)==1 else Node('row',children=tuple(children))

    def factor(self):
        self.depth+=1
        if self.depth>32: raise SyntaxError('Nesting exceeds 32 levels')
        token=self.take()
        if token in '+-−': node=Node('row',children=(Node('glyph','−' if token=='-' else token),self.factor()))
        elif token in '([{':
            closing={'(':')','[':']','{':'}'}[token]
            inner=self.sequence()
            if self.take()!=closing: raise SyntaxError('Mismatched grouping delimiter')
            node=Node('group','()' if token=='{' else token+closing,(inner,))
        elif token in '/^_*=×÷⇒→≤≥≠±)]}': raise SyntaxError('Expected an operand, got '+token)
        else: node=Node('glyph',token)
        scripts={}
        while self.peek() and self.peek() in '^_':
            which=self.take()
            if which in scripts: raise SyntaxError('Duplicate script; use braces for compound scripts')
            # Unbraced scripts consume exactly ONE symbol, matching conventional math notation.
            if self.peek()=='{':
                self.take();script=self.sequence()
                if self.take()!='}': raise SyntaxError('Unclosed script')
            else:
                char=self.take()
                if char not in LABELS or char in '+-−=×÷⇒()[]': raise SyntaxError('Script requires a symbol or braced expression')
                script=Node('glyph',char)
            scripts[which]=script
        if scripts: node=Node('script',children=(node,scripts.get('^'),scripts.get('_')))
        self.depth-=1;return node

def parse(text): return Parser(text).parse()

def required(node):
    result=set()
    if node.kind=='glyph': result.add(node.value)
    if node.kind=='fraction': result.add('fraction_bar')
    if node.kind=='group' and node.value!='{}':result.update(node.value)
    for child in node.children:
        if child is not None: result.update(required(child))
    return result

def tex(node):
    if node.kind=='glyph':return SYMBOLS.get(node.value,node.value.replace('−','-'))+' '
    if node.kind=='row':return ''.join(tex(child) for child in node.children)
    if node.kind=='group':return ('{'+tex(node.children[0])+'}' if node.value=='{}'
                                  else node.value[0]+tex(node.children[0])+node.value[1])
    if node.kind=='fraction':return r'\frac{'+tex(node.children[0])+'}{'+tex(node.children[1])+'}'
    if node.kind=='script':
        base,sup,sub=node.children
        return '{'+tex(base)+'}'+('_{'+tex(sub)+'}' if sub else '')+('^{'+tex(sup)+'}' if sup else '')
    raise ValueError(node.kind)
