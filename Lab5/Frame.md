?- define_class(animal, nil, [slot(color, unknown), slot(size, medium)]).
true.

?- define_class(bird, animal, [slot(can_fly, yes), slot(sound, chirp)]).
true.

?- define_class(parrot, bird, [slot(can_talk, yes)]).
true.

?- create_instance(polly, parrot).
true.

?- create_instance(tweety, bird).
true.

?- set_slot(polly, color, green).
true.

?- set_slot(tweety, size, small).
true.

?- get_slot(polly, color, X).
X = green.

?- get_slot(polly, can_fly, X).
X = yes.

?- get_slot(polly, size, X).
X = medium.

?- get_slot(tweety, sound, X).
X = chirp.

?- all_slots(parrot, Slots).
Slots = [slot(can_talk, yes), slot(can_fly, yes), slot(sound, chirp), slot(color, unknown), slot(size, medium)].
