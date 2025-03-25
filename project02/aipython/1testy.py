from stripsProblem import *
from stripsForwardPlanner import Forward_STRIPS
from searchMPP import SearcherMPP
import time


def h_blocks_distance(state, goal):
    return sum(1 for key in goal if state.get(key) != goal[key])


def get_actions_from_path(path):
    actions = []
    current = path
    while current.arc is not None:
        actions.append(current.arc.action)
        current = current.initial
    actions.reverse()
    return actions


blocks = {'a', 'b', 'c'}
tables = {'t1', 't2', 't3'}
all_places = blocks | tables
boolean = {True, False}


def on(x): return f"{x}_on"
def clear(x): return f"clear_{x}"


actions = set()

# move actions
for b in blocks:
    for x in all_places:
        for y in all_places:
            if x != y and b != x and b != y:
                actions.add(Strips(
                    name=f"move_{b}_from_{x}_to_{y}",
                    preconds={on(b): x, clear(b): True, clear(y): True},
                    effects={on(b): y, clear(x): True, clear(y): False}
                ))

feature_domain_dict = {on(b): all_places - {b} for b in blocks}
feature_domain_dict.update({clear(p): boolean for p in all_places})
blocks_domain = STRIPS_domain(feature_domain_dict, actions)

initial_state = {
    on('c'): 't1',
    on('b'): 'c',
    on('a'): 'b',
    clear('a'): True,
    clear('t2'): True,
    clear('t3'): True,
    clear('b'): False,
    clear('c'): False,
    clear('t1'): False
}

goal_state = {
    on('a'): 'b',
    on('b'): 'c',
    on('c'): 't2'
}

problem = Planning_problem(blocks_domain, initial_state, goal_state)


print("Uruchamiam planowanie z heurystyką...\n")
start = time.time()
solution = SearcherMPP(Forward_STRIPS(problem, h_blocks_distance)).search()
end = time.time()


if solution:
    actions = get_actions_from_path(solution)
    print("Znalezione działania:")
    for i, action in enumerate(actions, 1):
        print(f"{i}. {action}")
    print(f"\nCzas rozwiązania: {end - start:.4f} sekundy")
    print(f"Liczba instancji akcji: {len(actions)}")
else:
    print("Nie znaleziono rozwiązania.")
