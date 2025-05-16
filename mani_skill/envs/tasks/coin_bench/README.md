# Envs

This is the description  of COIN Bench
There are totally two parts of tasks in this bench

1. Primitive actions
2. Interactive reasoning
And the name system are following like this
coin-[actions]-[objects]-[description]-[version]
## actions
### primitive actions
Actions could be divided into some specifics
- pick 
- put(pick+place)
- open
- close 
- lift 
- rotata
- stack 



### interactive actions 
- find 

Objects will depend on the tasks
Description are somethings like

- [into cabinet]
- [into hole]
Blabla


## Environment Table

| Name                                   | Class       | Scene Building | Reward Building | Dataset Collection | Image                                                                                               | Video                                                                                                            |
| -------------------------------------- | ----------- | -------------- | --------------- | ------------------ | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Tabletop-Close-Door-v1                 | Primitive   | [x]            | [ ]             | [ ]                | ![Tabletop-Close-Door-v1](medias/images/material_0.png)                                             | <video src="medias/videos/Tabletop_Close_Door_v1_20250327_132429.mp4" width="320" height="240" controls></video> |
| Tabletop-Close-Cabinet-v1              | Primitive   | [x]            | [ ]             | [ ]                | ![Tabletop-Close-Cabinet-v1](medias/images/Tabletop-Close-Cabinet-v1.png)                           | <video src="medias/videos/Tabletop-Close-Cabinet-v1.mp4" width="320" height="240" controls></video>              |
| Tabletop-Find-Seal-v1                  | Interactive | [x]            | [ ]             | [ ]                | ![Tabletop-Find-Seal-v1](medias/images/Tabletop-Find-Seal-v1.png)                                   | <video src="medias/videos/Tabletop-Find-Seal-v1.mp4" width="320" height="240" controls></video>                  |
| Tabletop-Insert-Objects-v1             | Interactive | [x]            | [ ]             | [ ]                | ![Tabletop-Insert-Objects-v1](medias/images/Tabletop-Insert-Objects-v1.png)                         | <video src="medias/videos/Tabletop-Insert-Objects-v1.mp4" width="320" height="240" controls></video>             |
| Tabletop-Lift-Book-v1                  | Interactive | [x]            | [ ]             | [ ]                | ![Tabletop-Lift-Book-v1](medias/images/Tabletop-Lift-Book-v1.png)                                   | <video src="medias/videos/Tabletop-Lift-Book-v1.mp4" width="320" height="240" controls></video>                  |
| Tabletop-Move-Apple-DynamicFriction-v1 | Interactive | [x]            | [ ]             | [ ]                | ![Tabletop-Move-Apple-DynamicFriction-v1](medias/images/Tabletop-Move-Apple-DynamicFriction-v1.png) | <video src="medias/videos/Tabletop-Move-Apple-DynamicFriction-v1.mp4" width="320" height="240" controls></video> |
| Tabletop-Move-Apple-DynamicMass-v1     | Interactive | [x]            | [ ]             | [ ]                | ![Tabletop-Move-Apple-DynamicMass-v1](medias/images/Tabletop-Move-Apple-DynamicMass-v1.png)         | <video src="medias/videos/Tabletop-Move-Apple-DynamicMass-v1.mp4" width="320" height="240" controls></video>     |
| Tabletop-Move-Balls-WithPivot-v1       | Interactive | [x]            | [ ]             | [ ]                | ![Tabletop-Move-Balls-WithPivot-v1](medias/images/Tabletop-Move-Balls-WithPivot-v1.png)             | <video src="medias/videos/Tabletop-Move-Balls-WithPivot-v1.mp4" width="320" height="240" controls></video>       |
| Tabletop-Move-Cube-DynamicFriction-v1  | Interactive | [x]            | [ ]             | [ ]                | ![Tabletop-Move-Cube-DynamicFriction-v1](medias/images/Tabletop-Move-Cube-DynamicFriction-v1.png)   | <video src="medias/videos/Tabletop-Move-Cube-DynamicFriction-v1.mp4" width="320" height="240" controls></video>  |
| Tabletop-Move-Cube-DynamicMass-v1      | Interactive | [x]            | [ ]             | [ ]                | ![Tabletop-Move-Cube-DynamicMass-v1](medias/images/Tabletop-Move-Cube-DynamicMass-v1.png)           | <video src="medias/videos/Tabletop-Move-Cube-DynamicMass-v1.mp4" width="320" height="240" controls></video>      |
| Tabletop-Move-Cube-WithPivot-v1        | Interactive | [x]            | [ ]             | [ ]                | ![Tabletop-Move-Cube-WithPivot-v1](medias/images/Tabletop-Move-Cube-WithPivot-v1.png)               | <video src="medias/videos/Tabletop-Move-Cube-WithPivot-v1.mp4" width="320" height="240" controls></video>        |
| Tabletop-Open-Cabinet-WithSwitch-v1    | Interactive | [x]            | [ ]             | [ ]                | ![Tabletop-Open-Cabinet-WithSwitch-v1](medias/images/Tabletop-Open-Cabinet-WithSwitch-v1.png)       | <video src="medias/videos/Tabletop-Open-Cabinet-WithSwitch-v1.mp4" width="320" height="240" controls></video>    |
| Tabletop-Open-Cabinet-v1               | Primitive   | [x]            | [ ]             | [ ]                | ![Tabletop-Open-Cabinet-v1](medias/images/Tabletop-Open-Cabinet-v1.png)                             | <video src="medias/videos/Tabletop-Open-Cabinet-v1.mp4" width="320" height="240" controls></video>               |
| Tabletop-Open-Door-v1                  | Primitive   | [x]            | [ ]             | [ ]                | ![Tabletop-Open-Door-v1](medias/images/Tabletop-Open-Door-v1.png)                                   | <video src="medias/videos/Tabletop-Open-Door-v1.mp4" width="320" height="240" controls></video>                  |
| Tabletop-Open-Microwave-v1             | Primitive   | [x]            | [ ]             | [ ]                | ![Tabletop-Open-Microwave-v1](medias/images/Tabletop-Open-Microwave-v1.png)                         | <video src="medias/videos/Tabletop-Open-Microwave-v1.mp4" width="320" height="240" controls></video>             |
| Tabletop-Open-Trigger-v1               | Primitive   | [x]            | [ ]             | [ ]                | ![Tabletop-Open-Trigger-v1](medias/images/Tabletop-Open-Trigger-v1.png)                             | <video src="medias/videos/Tabletop-Open-Trigger-v1.mp4" width="320" height="240" controls></video>               |
| Tabletop-Pick-Apple-v1                 | Primitive   | [x]            | [ ]             | [ ]                | ![Tabletop-Pick-Apple-v1](medias/images/Tabletop-Pick-Apple-v1.png)                                 | <video src="medias/videos/Tabletop-Pick-Apple-v1.mp4" width="320" height="240" controls></video>                 |
| Tabletop-Pick-Cube-ToHolder-v1         | Primitive   | [x]            | [ ]             | [ ]                | ![Tabletop-Pick-Cube-ToHolder-v1](medias/images/Tabletop-Pick-Cube-ToHolder-v1.png)                 | <video src="medias/videos/Tabletop-Pick-Cube-ToHolder-v1.mp4" width="320" height="240" controls></video>         |
| Tabletop-Pick-Cylinder-WithObstacle-v1 | Interactive | [x]            | [ ]             | [ ]                | ![Tabletop-Pick-Cylinder-WithObstacle-v1](medias/images/Tabletop-Pick-Cylinder-WithObstacle-v1.png) | <video src="medias/videos/Tabletop-Pick-Cylinder-WithObstacle-v1.mp4" width="320" height="240" controls></video> |
| Tabletop-Pick-Object-FromCabinet-v1    | Interactive | [x]            | [ ]             | [ ]                | ![Tabletop-Pick-Object-FromCabinet-v1](medias/images/Tabletop-Pick-Object-FromCabinet-v1.png)       | <video src="medias/videos/Tabletop-Pick-Object-FromCabinet-v1.mp4" width="320" height="240" controls></video>    |
| Tabletop-Pick-Objects-InBox-v1         | Interactive | [x]            | [ ]             | [ ]                | ![Tabletop-Pick-Objects-InBox-v1](medias/images/Tabletop-Pick-Objects-InBox-v1.png)                 | <video src="medias/videos/Tabletop-Pick-Objects-InBox-v1.mp4" width="320" height="240" controls></video>         |
| Tabletop-Pick-Pen-v1                   | Primitive   | [x]            | [ ]             | [ ]                | ![Tabletop-Pick-Pen-v1](medias/images/Tabletop-Pick-Pen-v1.png)                                     | <video src="medias/videos/Tabletop-Pick-Pen-v1.mp4" width="320" height="240" controls></video>                   |
| Tabletop-Put-Ball-IntoContainer-v1     | Primitive   | [x]            | [ ]             | [ ]                | ![Tabletop-Put-Ball-IntoContainer-v1](medias/images/Tabletop-Put-Ball-IntoContainer-v1.png)         | <video src="medias/videos/Tabletop-Put-Ball-IntoContainer-v1.mp4" width="320" height="240" controls></video>     |
| Tabletop-Put-Balls-IntoContainer-v1    | Interactive | [x]            | [ ]             | [ ]                | ![Tabletop-Put-Balls-IntoContainer-v1](medias/images/Tabletop-Put-Balls-IntoContainer-v1.png)       | <video src="medias/videos/Tabletop-Put-Balls-IntoContainer-v1.mp4" width="320" height="240" controls></video>    |
| Tabletop-Put-Cube-IntoMicrowave-v1     | Interactive | [x]            | [ ]             | [ ]                | ![Tabletop-Put-Cube-IntoMicrowave-v1](medias/images/Tabletop-Put-Cube-IntoMicrowave-v1.png)         | <video src="medias/videos/Tabletop-Put-Cube-IntoMicrowave-v1.mp4" width="320" height="240" controls></video>     |
| Tabletop-Put-Fork-OnPlate-v1           | Primitive   | [x]            | [ ]             | [ ]                | ![Tabletop-Put-Fork-OnPlate-v1](medias/images/Tabletop-Put-Fork-OnPlate-v1.png)                     | <video src="medias/videos/Tabletop-Put-Fork-OnPlate-v1.mp4" width="320" height="240" controls></video>           |
| Tabletop-Rotate-Cube-Twice-v1          | Interactive | [x]            | [ ]             | [ ]                | ![Tabletop-Rotate-Cube-Twice-v1](medias/images/Tabletop-Rotate-Cube-Twice-v1.png)                   | <video src="medias/videos/Tabletop-Rotate-Cube-Twice-v1.mp4" width="320" height="240" controls></video>          |
| Tabletop-Seek-Objects-WithObstacle-v1  | Interactive | [x]            | [ ]             | [ ]                | ![Tabletop-Seek-Objects-WithObstacle-v1](medias/images/Tabletop-Seek-Objects-WithObstacle-v1.png)   | <video src="medias/videos/Tabletop-Seek-Objects-WithObstacle-v1.mp4" width="320" height="240" controls></video>  |
| Tabletop-Stack-Cube-WithShape-v1       | Interactive | [x]            | [ ]             | [ ]                | ![Tabletop-Stack-Cube-WithShape-v1](medias/images/Tabletop-Stack-Cube-WithShape-v1.png)             | <video src="medias/videos/Tabletop-Stack-Cube-WithShape-v1.mp4" width="320" height="240" controls></video>       |
| Tabletop-Stack-Cubes-v0                | Primitive   | [x]            | [ ]             | [ ]                | ![Tabletop-Stack-Cubes-v0](medias/images/Tabletop-Stack-Cubes-v0.png)                               | <video src="medias/videos/Tabletop-Stack-Cubes-v0.mp4" width="320" height="240" controls></video>                |
| Tabletop-Stack-LongObjects-v1          | Interactive | [x]            | [ ]             | [ ]                | ![Tabletop-Stack-LongObjects-v1](medias/images/Tabletop-Stack-LongObjects-v1.png)                   | <video src="medias/videos/Tabletop-Stack-LongObjects-v1.mp4" width="320" height="240" controls></video>          |

