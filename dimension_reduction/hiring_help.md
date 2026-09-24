Logic for hiring help
- for each 5x5 region, let each helping hand take charge of one column. The earlier they get hired, the further away their designated column they get assigned. For example, the farmer will always get assigned to work at the leftmost column of the starting 5x5 region.
- if the origin tile of the 5x5 is occupied by the farmer/helper, move them further away from the origin along east or west direction until they reach their designated place.
- once the farmer reached their designated column, they can work on their farm tasks. once they have completed their farm task of the tile, they can move north/ south to a tile they have never worked on.
- No need to hire more helpers once every column has an assigned worker.
- When the available region expands, use the same method as above
