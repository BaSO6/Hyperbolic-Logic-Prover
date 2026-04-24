import re

class GoalStructure:
    def __init__(self, kind, domain, is_searchable=True):
        self.kind = kind       # equality, inequality, prop, logic
        self.domain = domain   # real, nat, complex, general
        self.is_searchable = is_searchable

    def __repr__(self):
        return f"Goal({self.kind}, {self.domain})"

class GoalClassifier:
    def __init__(self):
        self.ban_patterns = [
            r"^\d+\s*=\s*\d+$",
            r"^\d+\s*<\s*\d+$",
            r"Nat\.gcd\s+\d+\s+\d+",
        ]

    def classify(self, goal: str) -> GoalStructure:
        g = goal.strip()
        
        for pat in self.ban_patterns:
            if re.match(pat, g):
                return GoalStructure("trivial", "general", False)

        domain = "general"
        if "Real.log" in g or "Real.exp" in g or "ℝ" in g: domain = "real"
        elif "Nat.gcd" in g or "Nat.prime" in g or "ℕ" in g: domain = "nat"
        elif "Complex" in g or "ℂ" in g: domain = "complex"
        elif "ℚ" in g or "Rat" in g: domain = "rat"

        kind = "prop"
        if "∧" in g or "∨" in g or "↔" in g: kind = "logic"
        elif "=" in g: kind = "equality"
        elif "≤" in g or "<" in g or "≥" in g or ">" in g: kind = "inequality"
        elif "∃" in g or "Exists" in g: kind = "exists"
            
        return GoalStructure(kind, domain, True)