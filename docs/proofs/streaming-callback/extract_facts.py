#!/usr/bin/env python3
"""Extract Datalog facts from Rust source with tree-sitter (a real parser).

usage: uv run --with tree-sitter --with tree-sitter-rust extract_facts.py REPO FACTS_DIR

No fact is typed by hand. Unsupported shapes abort loudly instead of being skipped.
Facts (tab-separated, one relation per file):
  fn(name,file,line,in_test)      param(fn,pos,name)        call(caller,callee,line)
  arg_ident(caller,callee,line,pos,ident)   push(fn,receiver,line)
  let_call(fn,var,callee)         ctor(fn,type)             cap_arm(type,cap)
  requires(fn,cap)                guard(fn,line,guardfn,callee)   guard_arg(fn,line,guardfn,pos,ident)
  arg_line(caller,callee,callline,ident,identline)   exact line of each argument identifier
Methods in `impl .. for T` are named `T::method`.
"""
import re, sys, os
import tree_sitter_rust as tsr
from tree_sitter import Language, Parser

REPO, OUT = sys.argv[1], sys.argv[2]
G = "crates/larql-inference/src/layer_graph/generate"
FILES = [f"{G}/gpu/mod.rs", f"{G}/cpu.rs", f"{G}/mod.rs",
         "crates/larql-compute/src/cpu/mod.rs", "crates/larql-compute/src/lib.rs",
         "crates/larql-inference/src/vindex/kquant_forward/generation.rs"]
if os.environ.get("STREAMING_GATE_FILES"):  # test hook: parse these repo-relative files instead
    FILES = os.environ["STREAMING_GATE_FILES"].split(",")
P = Parser(Language(tsr.language()))
F = {k: [] for k in "fn param call arg_ident push let_call ctor cap_arm requires guard guard_arg arg_line".split()}

def txt(n): return n.text.decode()
def line(n): return n.start_point[0] + 1
def die(n, why): sys.exit(f"extract_facts: unsupported shape at line {line(n)}: {why}\n{txt(n)[:200]}")

def idents(n, acc=None):
    acc = [] if acc is None else acc
    if n.type == "identifier": acc.append(txt(n))
    for c in n.children: idents(c, acc)
    return acc

def ident_nodes(n, acc=None):
    acc = [] if acc is None else acc
    if n.type == "identifier": acc.append(n)
    for c in n.children: ident_nodes(c, acc)
    return acc

def callee_name(fn_node):
    if fn_node.type == "identifier": return txt(fn_node)
    if fn_node.type == "scoped_identifier": return txt(fn_node.child_by_field_name("name"))
    if fn_node.type == "field_expression": return txt(fn_node.child_by_field_name("field"))
    return None

def in_cfg_test(n):
    while n is not None:
        if n.type == "mod_item":
            prev = n.prev_named_sibling
            while prev is not None and prev.type == "attribute_item":
                if "cfg(test)" in txt(prev).replace(" ", ""): return True
                prev = prev.prev_named_sibling
        n = n.parent
    return False

def impl_prefix(n):
    while n is not None:
        if n.type == "impl_item" and n.child_by_field_name("trait") is not None:
            return txt(n.child_by_field_name("type")) + "::"
        n = n.parent
    return ""

def disjuncts(c):
    if c.type == "parenthesized_expression": return disjuncts(c.named_children[0])
    if c.type == "binary_expression" and txt(c.child_by_field_name("operator")) == "||":
        return disjuncts(c.child_by_field_name("left")) + disjuncts(c.child_by_field_name("right"))
    return [c]

def conjuncts(c):
    if c.type == "parenthesized_expression": return conjuncts(c.named_children[0])
    if c.type == "binary_expression":
        op = txt(c.child_by_field_name("operator"))
        if op == "&&": return conjuncts(c.child_by_field_name("left")) + conjuncts(c.child_by_field_name("right"))
        if op == "||": die(c, "`||` inside a capability conjunction")
    return [c]

def walk_fn(fn, name, path):
    params = fn.child_by_field_name("parameters")
    pnames = []
    for p in params.named_children:
        if p.type == "parameter":
            pnames.append(idents(p.child_by_field_name("pattern"))[0])
    for i, p in enumerate(pnames): F["param"].append((name, i, p))
    F["fn"].append((name, path, line(fn), int(in_cfg_test(fn))))
    body = fn.child_by_field_name("body")
    if body is None: return

    # `fn supports(&self, cap) -> bool { matches!(cap, A | B) }`  => capability arms
    if name.endswith("::supports"):
        for m in find(body, "macro_invocation"):
            if txt(m.child_by_field_name("macro")) == "matches":
                toks = re.sub(r"^[^,]*,", "", txt(m.child_by_field_name("macro").next_sibling) if False else txt(m), count=1)
                for cap in re.findall(r"(?:Capability::)?\b([A-Z][A-Za-z0-9]*)\b", toks.replace("matches!", "")):
                    if cap not in ("Capability",): F["cap_arm"].append((name.split("::")[0], cap))

    # `a.supports(Capability::X) && b.supports(Capability::Y)` returned from a fn => requires
    last = body.named_children[-1] if body.named_children else None
    if last is not None and last.type == "binary_expression" and not in_cfg_test(fn) and "supports(Capability::" in txt(last):
        for c in conjuncts(last):
            m = re.fullmatch(r"\s*\w+\.supports\(Capability::(\w+)\)\s*", txt(c))
            if not m: die(c, "non-trivial capability conjunct")
            F["requires"].append((name, m.group(1)))

    for n in find(body, "let_declaration"):
        v = n.child_by_field_name("value")
        if v is not None and v.type == "call_expression" and n.child_by_field_name("pattern").type == "identifier":
            cn = callee_name(v.child_by_field_name("function"))
            if cn: F["let_call"].append((name, txt(n.child_by_field_name("pattern")), cn))

    for n in find(body, "call_expression"):
        fnode = n.child_by_field_name("function"); cn = callee_name(fnode)
        if cn is None: continue
        args = n.child_by_field_name("arguments").named_children
        F["call"].append((name, cn, line(n)))
        for pos, a in enumerate(args):
            for node in ident_nodes(a):
                F["arg_ident"].append((name, cn, line(n), pos, txt(node)))
                F["arg_line"].append((name, cn, line(n), txt(node), line(node)))
        if fnode.type == "field_expression" and cn == "push":
            recv = fnode.child_by_field_name("value")
            if recv.type == "identifier": F["push"].append((name, txt(recv), line(n)))
        # Box::new(path::Type) => constructs Type
        if txt(fnode) == "Box::new" and args and args[0].type in ("scoped_identifier", "identifier"):
            t = callee_name(args[0])
            if t and t[0].isupper(): F["ctor"].append((name, t))

    # `if <disjunct> { ... return callee(...) ... }` with disjunct `!guardfn(args..)`
    for n in find(body, "if_expression"):
        cond = n.child_by_field_name("condition")
        rets = [r for r in find(n.child_by_field_name("consequence"), "return_expression")
                if r.named_children and r.named_children[0].type == "call_expression"]
        if not rets: continue
        callee = callee_name(rets[-1].named_children[0].child_by_field_name("function"))
        for d in disjuncts(cond):
            if d.type == "unary_expression" and txt(d).startswith("!") and d.named_children[0].type == "call_expression":
                gc = d.named_children[0]; g = callee_name(gc.child_by_field_name("function"))
                F["guard"].append((name, line(n), g, callee))
                for pos, a in enumerate(gc.child_by_field_name("arguments").named_children):
                    for i in idents(a): F["guard_arg"].append((name, line(n), g, pos, i))

def find(n, typ):
    out = []
    if n.type == typ: out.append(n)
    for c in n.children: out += find(c, typ)
    return out

for path in FILES:
    root = P.parse(open(os.path.join(REPO, path), "rb").read()).root_node
    for fn in find(root, "function_item"):
        walk_fn(fn, impl_prefix(fn) + txt(fn.child_by_field_name("name")), path)

os.makedirs(OUT, exist_ok=True)
for rel, rows in F.items():
    with open(os.path.join(OUT, rel + ".facts"), "w") as f:
        for r in sorted(set(rows)): f.write("\t".join(map(str, r)) + "\n")
print({k: len(set(v)) for k, v in F.items()})
