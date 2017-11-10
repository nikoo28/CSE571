all_members([H],L2) :- member(H,L2).
all_members([H|T],L2) :- member(H,L2), all_members(T, L2).
 
all_not([H]) :- not(H).
all_not([H|T]) :- not(H), all_not(T).
 
all_not_members([H],L2) :- not(member(H,L2)).
all_not_members([H|T],L2) :- not(member(H,L2)), all_not_members(T, L2).
 
and([H]) :- H.
and([H|T]) :- H, and(T).
or([H]) :- H,!.
or([H|_]) :- H,!.
or([_|T]) :- or(T).
 
solve(Mario, Luigi, Alfredo, Guiseppe, Nunzio) :-

    % all chefs
    Mario = [Mario_crust, Mario_cheese, Mario_meat, Mario_veggie],
    Luigi = [Luigi_crust, Luigi_cheese, Luigi_meat, Luigi_veggie],
    Alfredo = [Alfredo_crust, Alfredo_cheese, Alfredo_meat, Alfredo_veggie],
    Guiseppe = [Guiseppe_crust, Guiseppe_cheese, Guiseppe_meat, Guiseppe_veggie],
    Nunzio = [Nunzio_crust, Nunzio_cheese, Nunzio_meat, Nunzio_veggie],
    
    % grouping
    All = [Mario, Luigi, Alfredo, Guiseppe, Nunzio],
    
    % ensure all values exist once
    all_members([crisp, greek, sicilian, thin, whole_wheat], [Mario_crust, Luigi_crust, Alfredo_crust, Guiseppe_crust, Nunzio_crust]),
    all_members([gorgonzola, mozzarella, parmesan, provolone, romano], [Mario_cheese, Luigi_cheese, Alfredo_cheese, Guiseppe_cheese, Nunzio_cheese]),
    all_members([bacon, chicken, meatball, pepperoni, sausage], [Mario_meat, Luigi_meat, Alfredo_meat, Guiseppe_meat, Nunzio_meat]),
    all_members([broccoli, mushrooms, olives, onions, peppers], [Mario_veggie, Luigi_veggie, Alfredo_veggie, Guiseppe_veggie, Nunzio_veggie]),
    
    % clue 1
    % The pizza topped with onions (which wasn't made by Nunzio) 
    % didn't have a Greek crust. 
    % Alfredo's pizza didn't have a whole wheat crust and wasn't topped with pepperoni. 
    % No pizza had both olives and mozzarella.
    member([C1_crust,_,_,onions], All),
    not(Nunzio_veggie = onions),
    not(C1_crust = greek),
    not(Alfredo_crust = whole_wheat),
    not(Alfredo_meat = pepperoni),
    member([_,mozzarella,_,C1_veggie], All),
    not(C1_veggie = olives),
    
    % clue 2
    % Mario didn't make the bacon pizza (whose crust was neither crisp nor thin). 
    % The sausage-topped pizza (which didn't have a whole wheat crust) 
    % didn't include peppers. 
    % Luigi's pizza (which didn't have a crisp crust) wasn't topped with broccoli.
    member([C2_crust,_,bacon,_], All),
    not(Mario_meat = bacon),
    not(C2_crust = crisp),
    not(C2_crust = thin),
    member([C2_crust,_,sausage,C2_veggie], All),
    not(C2_crust = whole_wheat),
    not(C2_veggie = peppers),
    not(Luigi_veggie = broccoli),
    not(Luigi_crust = crisp),
    
    % clue 3
    % Alfredo's pizza (which didn't include chicken) didn't have a Sicilian crust. 
    % Neither the pizza with a whole wheat crust (which wasn't topped with onions) 
    % nor Nunzio's pizza was topped with provolone cheese. 
    % Neither the pie topped with broccoli nor the pepperoni pizza was the one that had a crisp crust.
    not(Alfredo_crust = sicilian),
    not(Alfredo_meat = chicken),
    not(Nunzio_cheese = provolone),
    member([whole_wheat,C3_cheese,_,C3_veggie], All),
    not(C3_veggie = onions),
    not(C3_cheese = provolone),
    member([C3_crust,_,pepperoni,_], All),
    not(C3_crust = crisp),
    member([C3_1_crust,_,_,broccoli], All),
    not(C3_1_crust = crisp),
    
    % clue 4
    % Neither Giuseppe's pizza nor the sausage pizza was the one topped with onions. 
    % The Sicilian pizza (which wasn't Luigi's) wasn't topped with peppers. 
    % Alfredo's pizza (which didn't include olives) wasn't topped with provolone cheese.
    not(Guiseppe_veggie = onions),
    member([_,_,sausage,C4_veggie], All),
    not(C4_veggie = onions),
    not(Luigi_crust = sicilian),
    member([sicilian,_,_,C4_1_veggie], All),
    not(C4_1_veggie = peppers),
    not(Alfredo_cheese = provolone),
    not(Alfredo_veggie = olives),
    
    % clue 5
    % Mario didn't make the pepperoni pizza. 
    % The sausage pizza (which wasn't made by Nunzio) 
    % wasn't topped with mushrooms. 
    % The broccoli pizza was neither the one with a thin crust 
    % nor the one topped with mozzarella.
    not(Mario_meat = pepperoni),
    not(Nunzio_meat = sausage),
    member([_,_,sausage,C5_veggie], All),
    not(C5_veggie = mushrooms),
    member([C5_crust,_,_,broccoli], All),
    member([_,C5_cheese,_,broccoli], All),
    not(C5_crust = thin),
    not(C5_cheese = mozzarella),
    
    % clue 6
    % The meatball pizza (which wasn't topped with onions) 
    % didn't have a crisp crust. 
    % The olive pizza wasn't topped with Romano. 
    % Luigi's pizza (which didn't have a thin crust) 
    % wasn't topped with provolone.
    member([C6_crust,_,meatball,C6_veggie], All),
    not(C6_veggie = onions),
    not(C6_crust = crisp),
    member([_,C6_cheese,_,olives], All),
    not(C6_cheese = romano),
    not(Luigi_cheese = provolone),
    not(Luigi_crust = thin),
    
    % clue 7
    % The pizza with a Greek crust wasn't topped with broccoli. 
    % The bacon pizza had neither Parmesan nor Romano cheeses. 
    % There were no peppers on the thin crust pizza.
    member([greek,_,_,C7_veggie], All),
    not(C7_veggie = broccoli),
    member([_,C7_cheese,bacon,_], All),
    not(C7_cheese = parmesan),
    not(C7_cheese = romano),
    member([thin,_,_,C7_1_veggie], All),
    not(C7_1_veggie = peppers).