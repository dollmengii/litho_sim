"""app.py 를 **코드 순서대로** 훑어 정의 전에 쓰는 이름을 찾는다."""
import ast, builtins, io, sys

SKIP = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef,
        ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def scan(node, known, out):
    """내포·람다 **안으로는 들어가지 않는다.** 그 안의 이름은 그 자리에서
    묶이는 것이라 바깥 순서와 상관이 없다."""
    if node is None or isinstance(node, SKIP):
        return
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) \
            and node.id not in known:
        out.append((node.lineno, node.id))
    for c in ast.iter_child_nodes(node):
        scan(c, known, out)


def stores(node, known):
    for x in ast.walk(node):
        if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store):
            known.add(x.id)
        elif isinstance(x, ast.alias):
            known.add((x.asname or x.name).split(".")[0])
        elif isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            known.add(x.name)


def step(node, known, out):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        known.add(node.name)
        return
    if isinstance(node, ast.Assign):
        scan(node.value, known, out)
        stores(node, known)
        return
    if isinstance(node, (ast.With, ast.If, ast.For, ast.While, ast.Try)):
        for f in ("test", "iter"):
            if getattr(node, f, None) is not None:
                scan(getattr(node, f), known, out)
        for it in getattr(node, "items", []):
            scan(it.context_expr, known, out)
            if it.optional_vars is not None:
                stores(it.optional_vars, known)
        if isinstance(node, ast.For):
            stores(node.target, known)
        for b in ("body", "orelse", "finalbody"):
            for s in getattr(node, b, []) or []:
                step(s, known, out)
        for h in getattr(node, "handlers", []):
            if h.name:
                known.add(h.name)
            for s in h.body:
                step(s, known, out)
        return
    scan(node, known, out)
    stores(node, known)


def check(path="app.py"):
    src = io.open(path, encoding="utf-8").read()
    known = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
    out = []
    for n in ast.parse(src).body:
        step(n, known, out)
    lines = src.splitlines()
    return [(ln, nm, lines[ln - 1].strip()[:70]) for ln, nm in out]


if __name__ == "__main__":
    bad = check(sys.argv[1] if len(sys.argv) > 1 else "app.py")
    for b in bad:
        print(" ", b)
    print("정의 전에 쓰는 이름 없음" if not bad else f"{len(bad)} 건")
