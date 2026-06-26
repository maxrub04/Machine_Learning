
:- discontiguous do/1.
:- dynamic player_at/1, item_at/2, holding/1, world_state/2, alive/0, game_won/0, dig_count/1, story_phase/1.

% Map
connected(meadow,n,forest_edge). connected(meadow,e,riverbank). connected(meadow,s,sandy_trail). connected(meadow,w,rocky_field).
connected(forest_edge,s,meadow). connected(forest_edge,n,deep_forest). connected(forest_edge,e,stream).
connected(deep_forest,s,forest_edge). connected(deep_forest,n,jungle). connected(deep_forest,w,cliff_base).
connected(jungle,s,deep_forest). connected(jungle,e,swamp). connected(jungle,n,volcano_slope).
connected(swamp,w,jungle). connected(swamp,n,ruins).
connected(volcano_slope,s,jungle). connected(volcano_slope,n,crater_rim).
connected(crater_rim,s,volcano_slope).
connected(riverbank,w,meadow). connected(riverbank,n,stream). connected(riverbank,s,beach).
connected(stream,s,riverbank). connected(stream,w,forest_edge). connected(stream,n,waterfall).
connected(waterfall,s,stream).
connected(beach,n,riverbank). connected(beach,e,cove). connected(beach,s,sea_shallows).
connected(sea_shallows,n,beach).
connected(cove,w,beach). connected(cove,e,cave_entrance).
connected(cave_entrance,w,cove). connected(cave_entrance,n,cave_interior).
connected(cave_interior,s,cave_entrance). connected(cave_interior,n,bunker_door).
connected(bunker_door,s,cave_interior).
connected(sandy_trail,n,meadow). connected(sandy_trail,e,beach). connected(sandy_trail,w,hill).
connected(hill,e,sandy_trail). connected(hill,n,rocky_field).
connected(rocky_field,e,meadow). connected(rocky_field,s,hill). connected(rocky_field,w,cliff_base).
connected(cliff_base,e,rocky_field). connected(cliff_base,n,deep_forest).
connected(ruins,s,swamp). connected(ruins,e,old_bridge).
connected(old_bridge,w,ruins). connected(old_bridge,n,missile_silo).
connected(missile_silo,s,old_bridge).

% 2D grid coordinates for ASCII map (row, col) - 7 rows x 8 cols
location_pos(crater_rim,    0,2). location_pos(missile_silo,  0,6).
location_pos(volcano_slope, 1,2). location_pos(ruins,         1,5). location_pos(old_bridge,    1,6).
location_pos(jungle,        2,2). location_pos(swamp,         2,3).
location_pos(deep_forest,   3,2). location_pos(stream,        3,4). location_pos(waterfall,     3,5).
location_pos(cliff_base,    4,1). location_pos(forest_edge,   4,2). location_pos(riverbank,     4,4).
location_pos(beach,         4,5). location_pos(cove,          4,6). location_pos(cave_entrance, 4,7).
location_pos(rocky_field,   5,1). location_pos(meadow,        5,2). location_pos(sandy_trail,   5,3).
location_pos(sea_shallows,  5,5). location_pos(cave_interior, 5,7).
location_pos(hill,          6,1). location_pos(bunker_door,   6,7).

map_abbr(crater_rim,cr).    map_abbr(missile_silo,ms).  map_abbr(volcano_slope,vs).
map_abbr(ruins,ru).         map_abbr(old_bridge,br).    map_abbr(jungle,jn).
map_abbr(swamp,sw).         map_abbr(deep_forest,df).   map_abbr(stream,st).
map_abbr(waterfall,wf).     map_abbr(cliff_base,cb).    map_abbr(forest_edge,fe).
map_abbr(riverbank,rb).     map_abbr(beach,bc).         map_abbr(cove,cv).
map_abbr(cave_entrance,ce). map_abbr(rocky_field,rf).   map_abbr(meadow,md).
map_abbr(sandy_trail,tr).   map_abbr(sea_shallows,ss).  map_abbr(cave_interior,ci).
map_abbr(hill,hl).          map_abbr(bunker_door,bn).

% Location descriptions
desc(meadow)        :- format("MEADOW~nParachute silk tangled in the weeds. Ash and salt in the air.~n").
desc(forest_edge)   :- format("FOREST EDGE~nTrees close in here. Gnarled trunks, path continuing north.~n").
desc(deep_forest)   :- format("DEEP FOREST~nCan't see the sky. Roots cross the ground everywhere. Something is off about how quiet it is.~n").
desc(jungle)        :- format("JUNGLE~nFerns chest-high, vines blocking everything. Hot and wet.~n").
desc(swamp)         :- format("SWAMP~nBrown water up to your ankles. Dead trees in the mud.~n").
desc(volcano_slope) :- format("VOLCANO SLOPE~nBlack rock, warm underfoot. Steam coming up through cracks.~n").
desc(crater_rim)    :- format("CRATER RIM~nYou can see the whole island from up here. Orange glow from the crater below.~n").
desc(riverbank)     :- format("RIVERBANK~nThe river is wide and running fast. No crossing here.~n").
desc(stream)        :- format("STREAM~nClear water, cold. Moss on the rocks.~n").
desc(waterfall)     :- format("WATERFALL~nWater drops off a ledge into a pool. Something dark behind the curtain.~n").
desc(beach)         :- format("BEACH~nGrey sand, driftwood, rusted metal half-buried. Old wreckage.~n").
desc(sea_shallows)  :- format("SEA SHALLOWS~nWater up to your waist. Gets deep fast past here. Don't go further.~n").
desc(cove)          :- format("COVE~nRocky inlet. Dark water, not moving.~n").
desc(cave_entrance) :- format("CAVE ENTRANCE~nA gap in the cliff. Cold air from inside. Scratch marks on the rock next to it.~n").
desc(cave_interior) :- format("CAVE INTERIOR~nWide chamber, dripping. Military crates along the walls. Steel door north, locked.~n").
desc(bunker_door)   :- format("BUNKER~nConcrete walls, generators running. Console in the middle of the room. Lights going.~n").
desc(sandy_trail)   :- format("SANDY TRAIL~nDust and dry scrub. Old boot prints half-covered by wind.~n").
desc(hill)          :- format("HILL~nGrassy slope. Metal sign at the top: DANGER - RESTRICTED ZONE. Barely readable.~n").
desc(rocky_field)   :- format("ROCKY FIELD~nSharp rock, hard ground. Nothing grows here.~n").
desc(cliff_base)    :- format("CLIFF BASE~nSheer rock, straight up. No way to climb it.~n").
desc(ruins)         :- format("RUINS~nOld concrete outpost, broken apart. Weeds through the floor. Ravine to the east with a bridge.~n").
desc(old_bridge)    :- format("OLD BRIDGE~nPlanks over a deep gap. Some are missing. It sways when you put weight on it.~n").
desc(missile_silo)  :- format("MISSILE SILO~nConcrete shaft wide as a house, open to the sky. Missile on the pad. PANDORA in stencil on the side. Fuel smell.~n").

% Items
initial_item(shovel,meadow). initial_item(key,waterfall). initial_item(rations,beach).
initial_item(planks,ruins). initial_item(wire_cutters,hill). initial_item(map_fragment,sandy_trail).
initial_item(radio,rocky_field). initial_item(gas_mask,swamp). initial_item(crowbar,cove).
initial_item(explosives,cave_interior). initial_item(flare,volcano_slope). initial_item(journal,ruins).
initial_item(compass,rocky_field). initial_item(rope,waterfall).

% Item descriptions
idesc(shovel)       :- format("Folding shovel. Handle is bent but it digs fine.~n").
idesc(key)          :- format("Steel key, rusted. Serial number on the side.~n").
idesc(rations)      :- format("Tin of field rations. Still sealed.~n").
idesc(planks)       :- format("Heavy planks. Came from the outpost.~n").
idesc(wire_cutters) :- format("Wire cutters, insulated grips. Heavy.~n").
idesc(map_fragment) :- format("Piece of a torn map. East side of the island.~n").
idesc(radio)        :- format("Handheld radio. Just static.~n").
idesc(gas_mask)     :- format("Old gas mask, cracked visor. Seal still holds.~n").
idesc(crowbar)      :- format("Iron crowbar. Good heft.~n").
idesc(explosives)   :- format("C4 block, detonator attached. Don't drop it.~n").
idesc(flare)        :- format("Signal flare. One shot.~n").
idesc(journal)      :- format("Battered notebook. Handwriting is rushed, hard to read in places.~n").
idesc(tin_box)      :- format("Small tin box from underground. Something shifts inside when you move it.~n").
idesc(compass)      :- format("Brass compass. Points north.~n").
idesc(rope)         :- format("Thirty meters of nylon rope, military spec.~n").

% Init
init :-
    retractall(player_at(_)), retractall(item_at(_,_)), retractall(holding(_)),
    retractall(world_state(_,_)), retractall(alive), retractall(game_won),
    retractall(dig_count(_)), retractall(story_phase(_)),
    assert(player_at(meadow)), assert(alive), assert(dig_count(0)), assert(story_phase(arrival)),
    forall(initial_item(I,L), assert(item_at(I,L))).

show_map :-
    player_at(Here),
    holding(map_fragment), !,
    nl,
    write('  MAP    0   1   2   3   4   5   6   7'), nl,
    write('       +---+---+---+---+---+---+---+---+'), nl,
    forall(between(0, 6, Row),
        ( format("  ~w    |", [Row]),
          forall(between(0, 7, Col), print_map_cell(Row, Col, Here)),
          nl, write('       +---+---+---+---+---+---+---+---+'), nl )),
    nl, write('  * = your position'), nl.
show_map :-
    write('You dont have a map fragment in your inventory.'), nl.

print_map_cell(Row, Col, Here) :-
    ( location_pos(Loc, Row, Col) ->
        map_abbr(Loc, Ab),
        ( Loc == Here -> format("*~w|", [Ab]) ; format(" ~w|", [Ab]) )
    ; write(' . |') ).

% Look
look :-
    player_at(Here), nl, desc(Here),
    show_state(Here), show_items(Here), show_exits(Here), nl.

show_state(meadow) :- world_state(meadow,deep_hole), !, write('A deep pit in the ground. Something metal at the bottom.'), nl.
show_state(meadow) :- world_state(meadow,dug), !, write('A hole where you dug.'), nl.
show_state(cave_interior) :- world_state(bunker_door,unlocked), !, write('The steel door is open.'), nl.
show_state(cave_interior) :- !, write('The steel door is locked.'), nl.
show_state(old_bridge) :- world_state(old_bridge,reinforced), !, write('You fixed it. It holds now.'), nl.
show_state(old_bridge) :- !, write('Gaps in the planks. Crossing looks bad.'), nl.
show_state(bunker_door) :- world_state(missile,sabotaged), !, write('Console wrecked. Countdown is dead.'), nl.
show_state(bunker_door) :- !, write('Clock on the wall: 02:17:44. Still running.'), nl.
show_state(missile_silo) :- world_state(missile,sabotaged), !, write('The missile is dead on its pad.'), nl.
show_state(missile_silo) :- !, write('Guidance system humming. It is live.'), nl.
show_state(_).

show_items(Loc) :-
    findall(Item, item_at(Item,Loc), Items),
    (Items \= [] -> atomic_list_concat(Items, ', ', Out), format("You see: ~w~n", [Out]) ; true).

show_exits(Loc) :-
    findall(Dir, connected(Loc,Dir,_), Dirs),
    atomic_list_concat(Dirs, ', ', Out), format("Exits: ~w~n", [Out]).

% Start / Loop
start :-
    init, nl,
    write(' P A N D O R A''S   B O X '), nl, nl,
    write('1962. The Cuban Missile Crisis never ended.'), nl,
    write('The bombs fell in October. Billions dead in weeks.'), nl, nl,
    write('Survivors call it the Second World War.'), nl,
    write('What is left of humanity is on the eastern coast of South America.'), nl,
    write('They call it Neo City.'), nl, nl,
    write('You are CIA agent.'), nl,
    write('Three days ago a shortwave signal came from a small island off Cuba.'), nl,
    write('A Soviet auto-launch system, codename PANDORA, survived the war.'), nl,
    write('Armed. Target: Neo City.'), nl, nl,
    write('You went in by helicopter at dawn.'), nl,
    write('Soviet AA opened fire fifteen kilometers out.'), nl,
    write('You bailed. The helicopter is gone.'), nl, nl,
    write('No extraction until you signal.'), nl,
    write('Find the missile. Kill it. Fire the flare.'), nl, nl,
    write('Commands: help.'), nl, nl,
    look, game_loop.

game_loop :-
    (alive -> nl, write('> '),
        catch(read(C), _, C = bad_input),
        (C == quit -> write('Mission aborted.'), nl ;
         do(C), game_loop)
    ; game_won -> true
    ; nl, write('GAME OVER. Type "start." to try again.'), nl).

% Commands
do(help) :- nl,
    write('  north/south/east/west (or n/s/e/w)  Move'), nl,
    write('  look (or l)       Look around'), nl,
    write('  map               Show island map'), nl,
    write('  take(Item)        Pick up'), nl,
    write('  drop(Item)        Put down'), nl,
    write('  inventory (or i)  What you are carrying'), nl,
    write('  examine(Item)     Look at item'), nl,
    write('  use(Item)         Use item'), nl,
    write('  dig               Dig (need shovel)'), nl,
    write('  eat               Eat rations'), nl,
    write('  read(Item)        Read item'), nl,
    write('  quit              Quit'), nl.

% Movement
do(north) :- move(n). do(south) :- move(s). do(east) :- move(e). do(west) :- move(w).
do(n) :- move(n). do(s) :- move(s). do(e) :- move(e). do(w) :- move(w).

move(Dir) :- player_at(Here), (connected(Here,Dir,There) -> try_move(Here,Dir,There) ; blocked_move(Here,Dir)).

% Lethal dead ends
blocked_move(crater_rim, Dir) :- Dir \= s, !,
    write('You go over the rim. Magma. That is it.'), nl, die.
blocked_move(sea_shallows, Dir) :- Dir \= n, !,
    write('Current takes you. Water over your head. Gone.'), nl, die.
blocked_move(waterfall, Dir) :- Dir \= s, !,
    write('You slip on wet rock behind the falls. Long drop.'), nl, die.
blocked_move(missile_silo, Dir) :- Dir \= s, !,
    write('Grating gives way. Thirty meters straight down.'), nl, die.
blocked_move(old_bridge, Dir) :- (Dir == s ; Dir == e), !,
    write('No railing. You go off the edge. Sixty meters.'), nl, die.

% Blocked directions with flavor
blocked_move(forest_edge, w)   :- !, write('Solid undergrowth. No way through.'), nl.
blocked_move(deep_forest, e)   :- !, write('Thorn bushes. No way through.'), nl.
blocked_move(jungle, w)        :- !, write('Too thick. No way in.'), nl.
blocked_move(swamp, s)         :- !, write('Deeper water south. Not going there.'), nl.
blocked_move(swamp, e)         :- !, write('Fallen trees piled up. No gap.'), nl.
blocked_move(volcano_slope, e) :- !, write('Rock drops off east. Bad edge.'), nl.
blocked_move(volcano_slope, w) :- !, write('Solidified lava wall. Solid.'), nl.
blocked_move(riverbank, e)     :- !, write('River is wide and fast here. No crossing.'), nl.
blocked_move(stream, e)        :- !, write('Rock face on the east bank.'), nl.
blocked_move(beach, w)         :- !, write('Cliff goes straight into the water.'), nl.
blocked_move(cove, n)          :- !, write('Vertical cliff. No holds.'), nl.
blocked_move(cove, s)          :- !, write('Deep water. No bottom visible.'), nl.
blocked_move(cave_entrance, s) :- !, write('Solid cliff. No gap.'), nl.
blocked_move(cave_entrance, e) :- !, write('Rock face, unbroken.'), nl.
blocked_move(cave_interior, e) :- !, write('Cave wall.'), nl.
blocked_move(cave_interior, w) :- !, write('Cave wall.'), nl.
blocked_move(bunker_door, n)   :- !, write('Reinforced concrete. Would take explosives.'), nl.
blocked_move(bunker_door, e)   :- !, write('Dead screens along that wall. No exit.'), nl.
blocked_move(bunker_door, w)   :- !, write('Generator bank. Bolted in. No gap.'), nl.
blocked_move(sandy_trail, s)   :- !, write('Thorny scrub. Need a machete.'), nl.
blocked_move(hill, s)          :- !, write('Steep drop south.'), nl.
blocked_move(hill, w)          :- !, write('Loose shale. No path.'), nl.
blocked_move(rocky_field, n)   :- !, write('Cliff face. Try the cliff base to the west.'), nl.
blocked_move(cliff_base, s)    :- !, write('Cliff runs south. Nowhere to go.'), nl.
blocked_move(cliff_base, w)    :- !, write('Solid rock.'), nl.
blocked_move(ruins, n)         :- !, write('North wall is rubble.'), nl.
blocked_move(ruins, w)         :- !, write('Old gate buried under concrete.'), nl.
blocked_move(_, _)             :- write('No way through.'), nl.

try_move(cave_interior,n,bunker_door) :- !,
    (world_state(bunker_door,unlocked) -> do_move(bunker_door)
    ; write('Locked. Find a key or something to break it with.'), nl).
try_move(old_bridge,n,missile_silo) :- !,
    (world_state(old_bridge,reinforced) -> do_move(missile_silo)
    ; write('Too many gaps. You will fall. Fix the planks first.'), nl).
try_move(volcano_slope,n,crater_rim) :- !,
    (holding(gas_mask) -> do_move(crater_rim)
    ; write('Sulfur fumes. You choke and turn back. Need a mask.'), nl).
try_move(_,_,There) :- do_move(There).

do_move(There) :- retract(player_at(_)), assert(player_at(There)), look, update_story(There).

update_story(cave_interior) :- story_phase(Phase), Phase == arrival, !,
    retract(story_phase(Phase)), assert(story_phase(bunker_found)),
    nl, write('* Getting closer. *'), nl.''
update_story(bunker_door) :- story_phase(Phase), Phase == bunker_found, !,
    retract(story_phase(Phase)), assert(story_phase(inside_bunker)),
    nl, write('* The bunker. This is the end of it, one way or another. *'), nl.
update_story(missile_silo) :- !,
    nl, write('* There it is. PANDORA. *'), nl.
update_story(_).

% Look / Inventory / Map
do(look) :- look. do(l) :- look.
do(map) :- show_map.
do(inventory) :- do(i).
do(i) :- findall(Item,holding(Item),Items),
    (Items\=[] -> atomic_list_concat(Items, ', ', Out), format("Carrying: ~w~n", [Out])
    ; write('Nothing on you.'), nl).

% Take / Drop
do(take(Item)) :- player_at(Here),
    (item_at(Item,Here) -> retract(item_at(Item,Here)), assert(holding(Item)), write('Taken.'), nl
    ; format("No ~w here.~n", [Item])).
do(drop(Item)) :- player_at(Here),
    (holding(Item) -> retract(holding(Item)), assert(item_at(Item,Here)), write('Dropped.'), nl
    ; write('You do not have that.'), nl).

% Examine
do(examine(Item)) :- (holding(Item) ; (player_at(Here), item_at(Item,Here))),
    (idesc(Item) -> true ; write('Nothing special.'), nl), !.
do(examine(Item)) :- format("No ~w here.~n", [Item]).

% Read
do(read(journal)) :- holding(journal), !,
    nl, write('"Day 14. Missile is real. Silo is north of the bridge.'), nl,
    write(' Bunker console runs the launch. Cut the cables or wreck it.'), nl,
    write(' Fumes stopped me at the volcano. Mask must be in the swamp."'), nl.
do(read(journal)) :- !, write('You do not have the journal.'), nl.
do(read(map_fragment)) :- holding(map_fragment), !,
    nl, write('Cave through the cove. Locked bunker door. Bridge to the silo.'), nl,
    write('Key marked near a waterfall.'), nl.
do(read(map_fragment)) :- !, write('You do not have the map fragment.'), nl.
do(read(Item)) :- format("Can not read the ~w.~n", [Item]).

% Eat
do(eat) :- (holding(rations) -> retract(holding(rations)),
    write('Salted cardboard. But you needed it.'), nl
    ; write('Nothing to eat.'), nl).

% Dig
do(dig) :- (holding(shovel) -> player_at(Here), dig_at(Here) ; write('No shovel.'), nl).

dig_at(meadow) :- dig_count(Count), Count >= 2, !,
    nl, write('You keep digging. Walls cave in. You are buried.'), nl, die.
dig_at(meadow) :- !,
    dig_count(Count), retract(dig_count(Count)), Next is Count+1, assert(dig_count(Next)),
    (world_state(meadow,dug) ->
        retract(world_state(meadow,dug)), assert(world_state(meadow,deep_hole)),
        write('Deeper. Something metal down there. A tin box.'), nl,
        assert(item_at(tin_box,meadow))
    ; assert(world_state(meadow,dug)), write('Hole dug. Nothing yet.'), nl).
dig_at(rocky_field) :- !, write('Shovel bounces off. Too hard.'), nl.
dig_at(volcano_slope) :- !, write('Crack opens. Steam blasts your hands. You drop.'), nl, die.
dig_at(beach) :- !,
    (world_state(beach,dug) -> write('Just sand.'), nl
    ; assert(world_state(beach,dug)),
      write('Dog tag. "SGT. Rex - RECON 7"'), nl).
dig_at(_) :- write('Nothing here.'), nl.

% Use items
do(use(Item)) :- (holding(Item) -> player_at(Here), use_item(Item,Here) ; write('You do not have that.'), nl).

% "Already done" guards - must come before the action clauses
use_item(key,          cave_interior) :- world_state(bunker_door,unlocked), !,
    write('The door is already open.'), nl.
use_item(crowbar,      cave_interior) :- world_state(bunker_door,unlocked), !,
    write('The door is already open.'), nl.
use_item(planks,       old_bridge)    :- world_state(old_bridge,reinforced), !,
    write('The bridge is already fixed.'), nl.
use_item(rope,         old_bridge)    :- world_state(old_bridge,reinforced), !,
    write('The bridge is already fixed.'), nl.
use_item(wire_cutters, bunker_door)   :- world_state(missile,sabotaged), !,
    write('Missile is already down.'), nl.
use_item(wire_cutters, missile_silo)  :- world_state(missile,sabotaged), !,
    write('Missile is already down.'), nl.
use_item(explosives,   bunker_door)   :- world_state(missile,sabotaged), !,
    write('Missile is already down.'), nl.
use_item(explosives,   missile_silo)  :- world_state(missile,sabotaged), !,
    write('Missile is already down.'), nl.
use_item(rations, _) :- !, do(eat).

% Action clauses
use_item(key, cave_interior) :- \+ world_state(bunker_door,unlocked), !,
    write('Key goes in. Click. Door opens.'), nl,
    assert(world_state(bunker_door,unlocked)).
use_item(crowbar, cave_interior) :- \+ world_state(bunker_door,unlocked), !,
    write('You lever the frame. Lock snaps. Door grinds open.'), nl,
    assert(world_state(bunker_door,unlocked)).
use_item(planks, old_bridge) :- \+ world_state(old_bridge,reinforced), !,
    write('Planks across the gaps. It will hold.'), nl,
    retract(holding(planks)), assert(world_state(old_bridge,reinforced)).
use_item(rope, old_bridge) :- \+ world_state(old_bridge,reinforced), !,
    write('You lash the rope across the gaps as a lifeline. It will hold.'), nl,
    retract(holding(rope)), assert(world_state(old_bridge,reinforced)).
use_item(wire_cutters, bunker_door) :- \+ world_state(missile,sabotaged), !,
    write('Every cable cut. Sparks. The clock stops.'), nl,
    assert(world_state(missile,sabotaged)).
use_item(wire_cutters, missile_silo) :- \+ world_state(missile,sabotaged), !,
    write('Fuel lines cut. Propellant drains out. Missile is dead.'), nl,
    assert(world_state(missile,sabotaged)).
use_item(explosives, bunker_door) :- \+ world_state(missile,sabotaged), !,
    write('C4 on the console. You run. BOOM. Clock is gone.'), nl,
    retract(holding(explosives)), assert(world_state(missile,sabotaged)).
use_item(explosives, missile_silo) :- \+ world_state(missile,sabotaged), !,
    write('C4 on the tank. You run. The silo blows. PANDORA is scrap.'), nl,
    retract(holding(explosives)), assert(world_state(missile,sabotaged)).
use_item(explosives, _) :- !,
    write('You set it off here. Bad idea.'), nl,
    retract(holding(explosives)), die.
use_item(flare, crater_rim) :- world_state(missile,sabotaged), !,
    nl, write('Flare up from the rim. Red light over the island.'), nl,
    write('Twenty minutes. Rotors. Helicopter from THE CITY.'), nl, nl,
    write(' MISSION COMPLETE '), nl,
    write('PANDORA is gone. You are going home.'), nl,
    write('The world gets another day.'), nl,
    retract(holding(flare)), win.
use_item(flare, crater_rim) :- !,
    nl, write('Flare up from the rim. Red light over the island.'), nl,
    write('Twenty minutes. Rotors. Helicopter from Neo City.'), nl, nl,
    write(' MISSION FAILED '), nl,
    write('You got out. PANDORA did not.'), nl,
    write('Three days later, Neo City is ash.'), nl,
    retract(holding(flare)), die.
use_item(flare, _) :- !,
    write('Flare fires and fizzles low. Save it for high ground.'), nl.
use_item(radio, _) :- !,
    (world_state(missile,sabotaged) ->
        write('"Agent, we have you. Coming in."'), nl
    ; write('Static. Finish the job first.'), nl).
use_item(gas_mask, _) :- !, write('Mask on. You can breathe now.'), nl.
use_item(compass, _) :- !, show_map.
use_item(tin_box, _) :- world_state(tin_box,opened), !,
    write('Already open. Empty.'), nl.
use_item(tin_box, _) :- !,
    assert(world_state(tin_box,opened)),
    nl, write('You pry the lid. Inside: a folded note.'), nl,
    write('"PANDORA bunker is under the cave north of the cove.'), nl,
    write(' Silo is north past the bridge. Gas mask in the swamp."'), nl.
use_item(Item, _) :- format("Not sure what to do with the ~w here.~n", [Item]).

% Death / Win
die :- retract(alive), nl, write('YOU ARE DEAD.'), nl.
win :- assert(game_won), retract(alive).

% Catch-all
do(_) :- write('Unknown command. Type "help."'), nl.
