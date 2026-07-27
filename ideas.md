Running List of Ideas

- Tune the action chunk when training another bc model from scratch. Moreover, increase the amount of history of frames or maybe also incorporate a step counter.
    - It would mess with the dagger based on starting in near violations, but this can be fixed by incorpoating the history of where the scout (policy) trajectory was before the near violation

- look into the a* branch. when ready, merge it and then retrain the entire pipeline with the a* pathing (flow loss and safety loss should collide less becasue the expert is alr as far away as possible from the obstacles)



Need to collect v7 data (a* and 30 action chunk)
Train BC on v7 data
DAgger with a*, review performance of pathing