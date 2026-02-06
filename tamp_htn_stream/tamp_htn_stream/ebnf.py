# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Set
import argparse
import json
import re
import sys


# =========================
# Grammar AST nodes
# =========================

@dataclass(frozen=True)
class Node:
    pass

@dataclass(frozen=True)
class Ref(Node):
    name: str

@dataclass(frozen=True)
class Lit(Node):
    text: str

@dataclass(frozen=True)
class Regex(Node):
    pattern: str

@dataclass(frozen=True)
class Seq(Node):
    parts: Tuple[Node, ...]

@dataclass(frozen=True)
class Alt(Node):
    options: Tuple[Node, ...]

@dataclass(frozen=True)
class Repeat(Node):
    node: Node
    min_times: int
    max_times: Optional[int]  # None => infinity


# =========================
# Grammar tokenizer
# =========================

@dataclass
class Tok:
    kind: str   # 'ID', 'STR', 'REGEX', 'SYM', 'EOF'
    value: str
    pos: int    # position in grammar text (char index)

class GrammarLexer:
    def __init__(self, text: str):
        self.text = text
        self.n = len(text)
        self.i = 0

    def _peek(self) -> str:
        return self.text[self.i] if self.i < self.n else ""

    def _starts(self, s: str) -> bool:
        return self.text.startswith(s, self.i)

    def _skip_ws_and_comments(self) -> None:
        while self.i < self.n:
            ch = self._peek()
            # whitespace
            if ch.isspace():
                self.i += 1
                continue
            # line comments: #... or //...
            if self._starts("#"):
                while self.i < self.n and self.text[self.i] != "\n":
                    self.i += 1
                continue
            if self._starts("//"):
                self.i += 2
                while self.i < self.n and self.text[self.i] != "\n":
                    self.i += 1
                continue
            # block comments /* ... */
            if self._starts("/*"):
                self.i += 2
                while self.i < self.n and not self._starts("*/"):
                    self.i += 1
                if self._starts("*/"):
                    self.i += 2
                continue
            break

    def next(self) -> Tok:
        self._skip_ws_and_comments()
        if self.i >= self.n:
            return Tok("EOF", "", self.i)

        start = self.i

        # multi-char symbols
        if self._starts("::="):
            self.i += 3
            return Tok("SYM", "::=", start)

        ch = self._peek()

        # single-char symbols
        if ch in "=;|()[]{}*+?":
            self.i += 1
            return Tok("SYM", ch, start)

        # regex: /.../  (supports escaping \/ )
        if ch == "/":
            self.i += 1
            pat = []
            while self.i < self.n:
                c = self.text[self.i]
                if c == "\\" and self.i + 1 < self.n:
                    pat.append(c)
                    pat.append(self.text[self.i + 1])
                    self.i += 2
                    continue
                if c == "/":
                    self.i += 1
                    return Tok("REGEX", "".join(pat), start)
                pat.append(c)
                self.i += 1
            raise SyntaxError(f"Unterminated regex starting at pos {start}")

        # string: "..." or '...'
        if ch in "\"'":
            quote = ch
            self.i += 1
            buf = []
            while self.i < self.n:
                c = self.text[self.i]
                if c == "\\" and self.i + 1 < self.n:
                    buf.append(self.text[self.i + 1])
                    self.i += 2
                    continue
                if c == quote:
                    self.i += 1
                    return Tok("STR", "".join(buf), start)
                buf.append(c)
                self.i += 1
            raise SyntaxError(f"Unterminated string starting at pos {start}")

        # identifier
        if re.match(r"[A-Za-z_]", ch):
            j = self.i + 1
            while j < self.n and re.match(r"[A-Za-z0-9_]", self.text[j]):
                j += 1
            ident = self.text[self.i:j]
            self.i = j
            return Tok("ID", ident, start)

        raise SyntaxError(f"Unexpected character in grammar at pos {start}: {repr(ch)}")


# =========================
# Grammar parser (EBNF subset)
# =========================

class GrammarParser:
    def __init__(self, text: str):
        self.lex = GrammarLexer(text)
        self.cur = self.lex.next()

    def _eat(self, kind: str, value: Optional[str] = None) -> Tok:
        if self.cur.kind != kind or (value is not None and self.cur.value != value):
            exp = f"{kind}" + (f"({value})" if value else "")
            got = f"{self.cur.kind}({self.cur.value})"
            raise SyntaxError(f"Grammar parse error at pos {self.cur.pos}: expected {exp}, got {got}")
        t = self.cur
        self.cur = self.lex.next()
        return t

    def _accept(self, kind: str, value: Optional[str] = None) -> Optional[Tok]:
        if self.cur.kind == kind and (value is None or self.cur.value == value):
            return self._eat(kind, value)
        return None

    def parse(self) -> Dict[str, Node]:
        rules: Dict[str, Node] = {}
        while self.cur.kind != "EOF":
            name_tok = self._eat("ID")
            if self._accept("SYM", "::=") is None:
                self._eat("SYM", "=")

            expr = self._parse_expr()

            self._eat("SYM", ";")

            if name_tok.value in rules:
                raise SyntaxError(f"Duplicate rule: {name_tok.value}")
            rules[name_tok.value] = expr

        if not rules:
            raise SyntaxError("No rules found in grammar.")
        return rules

    def _parse_expr(self) -> Node:
        # expr := seq ('|' seq)*
        seqs = [self._parse_seq()]
        while self._accept("SYM", "|"):
            seqs.append(self._parse_seq())

        # If multiple => Alt
        if len(seqs) == 1:
            return seqs[0]
        return Alt(tuple(seqs))

    def _parse_seq(self) -> Node:
        # seq := (factor)*
        parts: List[Node] = []
        while True:
            # stop tokens for sequence
            if self.cur.kind == "EOF":
                break
            if self.cur.kind == "SYM" and self.cur.value in ("|", ")", "]", "}", ";"):
                break
            parts.append(self._parse_factor())

        # allow epsilon
        return Seq(tuple(parts))

    def _parse_factor(self) -> Node:
        # factor := atom (postfix)?
        atom = self._parse_atom()

        # postfix quantifiers: ?, *, +
        if self.cur.kind == "SYM" and self.cur.value in ("?", "*", "+"):
            q = self.cur.value
            self._eat("SYM", q)
            if q == "?":
                return Repeat(atom, 0, 1)
            if q == "*":
                return Repeat(atom, 0, None)
            if q == "+":
                return Repeat(atom, 1, None)

        return atom

    def _parse_atom(self) -> Node:
        # atom := ID | STR | REGEX | '(' expr ')' | '[' expr ']' | '{' expr '}'
        if self.cur.kind == "ID":
            t = self._eat("ID")
            return Ref(t.value)

        if self.cur.kind == "STR":
            t = self._eat("STR")
            return Lit(t.value)

        if self.cur.kind == "REGEX":
            t = self._eat("REGEX")
            return Regex(t.value)

        if self._accept("SYM", "("):
            inner = self._parse_expr()
            self._eat("SYM", ")")
            return inner

        if self._accept("SYM", "["):
            inner = self._parse_expr()
            self._eat("SYM", "]")
            return Repeat(inner, 0, 1)

        if self._accept("SYM", "{"):
            inner = self._parse_expr()
            self._eat("SYM", "}")
            return Repeat(inner, 0, None)

        raise SyntaxError(f"Grammar parse error at pos {self.cur.pos}: unexpected token {self.cur.kind}({self.cur.value})")


# =========================
# Input parser (PEG-style with memoization)
# =========================

@dataclass
class ParseFailure:
    farthest: int
    expected: Set[str]

@dataclass
class ParseOk:
    pos: int
    tree: Any

Result = Union[ParseOk, ParseFailure]


class EBNFInterpreter:
    def __init__(self, rules: Dict[str, Node], start: str, text: str, skip_ws: bool = True):
        self.rules = rules
        self.start = start
        self.text = text
        self.n = len(text)
        self.skip_ws = skip_ws
        self.ws_re = re.compile(r"\s*") if skip_ws else None

        # memo for rule calls: (name, pos) -> Result
        self.memo: Dict[Tuple[str, int], Result] = {}
        self.call_stack: List[Tuple[str, int]] = []  # for left recursion detection

    def _skip(self, pos: int) -> int:
        if not self.skip_ws:
            return pos
        assert not self.ws_re is None
        m = self.ws_re.match(self.text, pos)
        return m.end() if m else pos

    def _merge_fail(self, a: ParseFailure, b: ParseFailure) -> ParseFailure:
        if b.farthest > a.farthest:
            return b
        if a.farthest > b.farthest:
            return a
        # same farthest: merge expected
        return ParseFailure(a.farthest, a.expected | b.expected)

    def _fail(self, pos: int, expected: str) -> ParseFailure:
        return ParseFailure(pos, {expected})

    def parse(self) -> Any:
        res = self._parse_rule(self.start, 0)
        if isinstance(res, ParseFailure):
            self._raise_input_error(res)
        assert isinstance(res, ParseOk)
        pos = self._skip(res.pos)
        if pos != self.n:
            # leftover input
            fail = ParseFailure(pos, {"<EOF>"})
            self._raise_input_error(fail)
        return res.tree

    def _raise_input_error(self, fail: ParseFailure) -> None:
        pos = fail.farthest
        line = self.text.count("\n", 0, pos) + 1
        col = pos - (self.text.rfind("\n", 0, pos) + 1) + 1
        snippet = self.text[pos:pos+50].splitlines()[0] if pos < self.n else ""
        exp = ", ".join(sorted(fail.expected))
        raise ValueError(f"Parse error at line {line}, col {col} (pos {pos}). Expected: {exp}. Near: {snippet!r}")

    def _parse_rule(self, name: str, pos: int) -> Result:
        if name not in self.rules:
            return self._fail(pos, f"<undefined rule {name}>")

        key = (name, pos)
        if key in self.memo:
            return self.memo[key]

        # left recursion detection
        if key in self.call_stack:
            return self._fail(pos, f"<left recursion in {name}>")

        self.call_stack.append(key)
        node = self.rules[name]
        res = self._parse_node(node, pos)
        self.call_stack.pop()

        if isinstance(res, ParseOk):
            wrapped = ParseOk(res.pos, {"type": "rule", "name": name, "children": res.tree})
            self.memo[key] = wrapped
            return wrapped

        # failure
        self.memo[key] = res
        return res

    def _parse_node(self, node: Node, pos: int) -> Result:
        if isinstance(node, Ref):
            return self._parse_rule(node.name, pos)

        if isinstance(node, Lit):
            p = self._skip(pos)
            if self.text.startswith(node.text, p):
                end = p + len(node.text)
                return ParseOk(end, {"type": "lit", "text": node.text})
            return self._fail(p, repr(node.text))

        if isinstance(node, Regex):
            p = self._skip(pos)
            # regex match at current position
            try:
                r = re.compile(node.pattern)
            except re.error as e:
                return self._fail(p, f"<bad regex /{node.pattern}/: {e}>")
            m = r.match(self.text, p)
            if m:
                return ParseOk(m.end(), {"type": "re", "pattern": node.pattern, "text": m.group(0)})
            return self._fail(p, f"/{node.pattern}/")

        if isinstance(node, Seq):
            cur_pos = pos
            children: List[Any] = []
            worst_fail: Optional[ParseFailure] = None

            for part in node.parts:
                r = self._parse_node(part, cur_pos)
                if isinstance(r, ParseFailure):
                    worst_fail = r if worst_fail is None else self._merge_fail(worst_fail, r)
                    return worst_fail
                children.append(r.tree)
                cur_pos = r.pos

            return ParseOk(cur_pos, {"type": "seq", "children": children})

        if isinstance(node, Alt):
            best_fail: Optional[ParseFailure] = None
            for opt in node.options:
                r = self._parse_node(opt, pos)
                if isinstance(r, ParseOk):
                    return ParseOk(r.pos, {"type": "alt", "chosen": r.tree})
                best_fail = r if best_fail is None else self._merge_fail(best_fail, r)
            return best_fail or self._fail(pos, "<alt>")

        if isinstance(node, Repeat):
            cur_pos = pos
            children: List[Any] = []
            count = 0

            # repeat until no progress
            while True:
                r = self._parse_node(node.node, cur_pos)
                if isinstance(r, ParseFailure):
                    break
                if r.pos == cur_pos:
                    # protect against infinite loops on epsilon
                    break
                children.append(r.tree)
                cur_pos = r.pos
                count += 1
                if node.max_times is not None and count >= node.max_times:
                    break

            if count < node.min_times:
                # report failure at current point with expectation
                p = self._skip(cur_pos)
                return self._fail(p, f"<repeat x{node.min_times}+>")

            return ParseOk(cur_pos, {"type": "repeat", "min": node.min_times, "max": node.max_times, "children": children})

        return self._fail(pos, "<unknown node>")


def children_of(node):
    t = node.get("type")
    if t == "rule":
        return [node["children"]]
    if t == "seq":
        return node.get("children", [])
    if t == "alt":
        return [node["chosen"]]
    if t == "repeat":
        return node.get("children", [])
    return []

class VisitorAbstract:
    def printIndent(self, depth, s):
        print('{}{}'.format('  '*depth, s))

    def visit(self, node, depth):
        verbose = False
        t = node.get("type")
        if t == 'rule':
            if verbose:
                self.printIndent(depth, 'type: {}, name: {}'.format(t, node.get('name')))
            assert len(children_of(node)) == 1
            args = self.visit(children_of(node)[0], depth+1)
            visit_method_name = 'visit_{}'.format(node.get('name'))
            if hasattr(self, visit_method_name):
                method = getattr(self, visit_method_name)
                return method(args)
            # else:
            return args
        elif t == 'lit':
            if verbose:
                self.printIndent(depth, 'type: {}, text: {}'.format(t, node.get('text')))
            assert len(children_of(node)) == 0
            return node.get('text')
        elif t == 're':
            if verbose:
                self.printIndent(depth, 'type: {}, pattern: {}, text: {}'.format(t, node.get('pattern'), node.get('text')))
            assert len(children_of(node)) == 0
            return node.get('text')
        elif t == 'seq':
            if verbose:
                self.printIndent(depth, 'type: {}, count: {}'.format(t, len(node.get("children", []))))
            pass
        elif t == 'alt':
            if verbose:
                self.printIndent(depth, 'type: {}, count: {}'.format(t, len(node.get("children", []))))
            pass
        elif t == 'repeat':
            if verbose:
                self.printIndent(depth, 'type: {}, count: {}'.format(t, len(node.get("children", []))))
            pass
        else:
            raise Exception(t)

        result = []
        for ch in children_of(node):
            result.append( self.visit(ch, depth+1) )
        return result

