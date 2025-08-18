% Taxonomy
?- add_rel(bird, isa, animal).
true.
?- add_rel(parrot, isa, bird).
true.

% Properties
?- add_prop(animal, moves, yes).
true.
?- add_prop(bird, can_fly, yes).
true.
?- add_prop(parrot, sound, squawk).
true.

% Instance-level node + override
?- add_rel(polly, isa, parrot).
true.
?- add_prop(polly, can_fly, no).   % override
true.

% Inheritance
?- has_prop(polly, sound, V).
V = squawk.

?- has_prop(polly, moves, V).
V = yes.

?- has_prop(polly, can_fly, V).
V = no.

% Ancestors/descendants
?- ancestors(polly, As).
As = [parrot, bird, animal].

?- descendant_of(animal, Ds).
Ds = [bird, parrot, polly].

% Relations & paths
?- related(bird, isa, animal).
true.
?- path(polly, animal, P).
P = [polly, parrot, bird, animal].

?- labeled_path(polly, animal, L).
L = [edge(polly, isa, parrot), edge(parrot, isa, bird), edge(bird, isa, animal)].
