from stripsProblem import *
from stripsForwardPlanner import Forward_STRIPS
from searchMPP import SearcherMPP
import time

# === Heurystyka: odległość do celu (tu liczba przejść border) ===
def h_magic_path(state, goal):
    loc_from = state['at_npc']
    loc_to = goal['at_npc']
    paths = {
        'town': ['field'],
        'field': ['town', 'castle'],
        'castle': ['field']
    }
    visited = set()
    frontier = [(loc_from, 0)]
    while frontier:
        loc, dist = frontier.pop(0)
        if loc == loc_to:
            return dist
        visited.add(loc)
        for neighbor in paths.get(loc, []):
            if neighbor not in visited:
                frontier.append((neighbor, dist + 1))
    return 100  # default high cost if unreachable

# === Funkcja pomocnicza do wyciągania akcji z Path ===
def get_actions_from_path(path):
    actions = []
    current = path
    while current.arc is not None:
        actions.append(current.arc.action)
        current = current.initial
    actions.reverse()
    return actions

# === Definicja dziedziny Magic World (ograniczona do 'move') ===
boolean = {True, False}
locations = {'town', 'field', 'castle'}

feature_domain = {
    'at_npc': locations,
    'guarded_field': boolean,
    'guarded_castle': boolean
}

actions = set()

# move actions
borders = [('town', 'field'), ('field', 'castle')]
for (l1, l2) in borders:
    for (from_loc, to_loc) in [(l1, l2), (l2, l1)]:
        actions.add(Strips(
            name=f"move_npc_from_{from_loc}_to_{to_loc}",
            preconds={'at_npc': from_loc, f'guarded_{to_loc}': False},
            effects={'at_npc': to_loc}
        ))

magic_domain = STRIPS_domain(feature_domain, actions)

# === Definicja problemu ===
initial_state = {
    'at_npc': 'town',
    'guarded_field': False,
    'guarded_castle': False
}

goal_state = {
    'at_npc': 'castle'
}

magic_problem = Planning_problem(magic_domain, initial_state, goal_state)

# === Uruchomienie planowania z heurystyką ===
print("🔮 Uruchamiam Magic World planning z heurystyką...\n")
start = time.time()
solution = SearcherMPP(Forward_STRIPS(magic_problem, h_magic_path)).search()
end = time.time()

# === Wypisanie rozwiązania ===
if solution:
    actions = get_actions_from_path(solution)
    print("✅ Znalezione działania:")
    for i, action in enumerate(actions, 1):
        print(f"{i}. {action}")
    print(f"\n⏱️ Czas rozwiązania: {end - start:.4f} sekundy")
    print(f"🔢 Liczba akcji: {len(actions)}")
else:
    print("❌ Nie znaleziono rozwiązania.")
