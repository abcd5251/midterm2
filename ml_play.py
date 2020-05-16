"""
The template of the script for the machine learning process in game pingpong
"""

# Import the necessary modules and classes
from mlgame.communication import ml as comm
import os.path as path
import pickle
import numpy as np
def ml_loop(side: str):
    """
    The main loop for the machine learning process
    The `side` parameter can be used for switch the code for either of both sides,
    so you can write the code for both sides in the same script. Such as:
    ```python
    if side == "1P":
        ml_loop_for_1P()
    else:
        ml_loop_for_2P()
    ```
    @param side The side which this script is executed for. Either "1P" or "2P".
    """

    # === Here is the execution order of the loop === #
    # 1. Put the initialization code here
    ball_served = False
    ball_x=[]
    ball_y=[]
    ball_speed_x=[]
    ball_speed_y=[]
    brick_x=[]
    brick_y=[]
    platform1p=[]
    platform2p=[]
    ball_willbe1p_x=[]
    ball_willbe2p_x=[]
    command_1p=[]
    ball_pred=[]
    i=0
    k=0
    filename = path.join(path.dirname(__file__), 'save', 'prediction_1p.pickle')
    with open(filename, 'rb') as file:
       clf_1p = pickle.load(file)

    stop=False
    def move_to(player, pred) : #move platform to predicted position to catch ball 
        if player == '1P':
            if scene_info["platform_1P"][0]+20  > (pred-10) and scene_info["platform_1P"][0]+20 < (pred+10): return 0 # NONE
            elif scene_info["platform_1P"][0]+20 <= (pred-10) : return 1 # goes right
            else : return 2 # goes left
        else :
            if scene_info["platform_2P"][0]+20  > (pred-10) and scene_info["platform_2P"][0]+20 < (pred+10): return 0 # NONE
            elif scene_info["platform_2P"][0]+20 <= (pred-10) : return 1 # goes right
            else : return 2 # goes left

    def ml_loop_for_1P(): 
        if scene_info["ball_speed"][1] > 0 : # 球正在向下 # ball goes down
            x = ( scene_info["platform_1P"][1]-scene_info["ball"][1] ) // scene_info["ball_speed"][1] # 幾個frame以後會需要接  # x means how many frames before catch the ball
            pred = scene_info["ball"][0]+(scene_info["ball_speed"][0]*x)  # 預測最終位置 # pred means predict ball landing site 
            bound = pred // 200 # Determine if it is beyond the boundary
            if (bound > 0): # pred > 200 # fix landing position
                if (bound%2 == 0) : 
                    pred = pred - bound*200                    
                else :
                    pred = 200 - (pred - 200*bound)
            elif (bound < 0) : # pred < 0
                if (bound%2 ==1) :
                    pred = abs(pred - (bound+1) *200)
                else :
                    pred = pred + (abs(bound)*200)
            ball_pred.append(pred)
            return move_to(player = '1P',pred = pred)
        else:
            pred=100
            
            return move_to(player = '1P',pred = 100)
        """else : # 球正在向上 # ball goes up
            time=(scene_info["ball"][1]-260)//abs(scene_info["ball_speed"][1])
            pred = scene_info["ball"][0]+(scene_info["ball_speed"][0]*time)
            bound = pred // 200 # Determine if it is beyond the boundary
            if (bound > 0): # pred > 200 # fix landing position
                    if (bound%2 == 0) : 
                        pred = pred - bound*200                    
                    else :
                        pred = 200 - (pred - 200*bound)
            elif (bound < 0) : # pred < 0
                    if (bound%2 ==1) :
                        pred = abs(pred - (bound+1) *200)
                    else :
                        pred = pred + (abs(bound)*200)
            if pred<=scene_info["blocker"][0]+3*time+10 or pred>=scene_info["blocker"][0]+3*time or pred<=scene_info["blocker"][0]-3*time+10 or pred>=scene_info["blocker"][0]-3*time: 
                x = 155//abs(scene_info["ball_speed"][1]*1.3)
                pred = scene_info["ball"][0]+(scene_info["ball_speed"][0]*x)
                bound = pred // 200 # Determine if it is beyond the boundary
                if (bound > 0): # pred > 200 # fix landing position
                    if (bound%2 == 0) : 
                        pred = pred - bound*200                    
                    else :
                        pred = 200 - (pred - 200*bound)
                elif (bound < 0) : # pred < 0
                    if (bound%2 ==1) :
                        pred = abs(pred - (bound+1) *200)
                    else :
                        pred = pred + (abs(bound)*200)
                return move_to(player = '1P',pred = pred)"""
           

    def ml_loop_for_2P():  # as same as 1P
        if scene_info["ball_speed"][1] >=0 : 
            pred=100
            
            return move_to(player = '2P',pred = 100)
        else : 
            x = ( scene_info["platform_2P"][1]+30-scene_info["ball"][1] ) // scene_info["ball_speed"][1] 
            pred = scene_info["ball"][0]+(scene_info["ball_speed"][0]*x) 
            bound = pred // 200 
            if (bound > 0):
                if (bound%2 == 0):
                    pred = pred - bound*200 
                else :
                    pred = 200 - (pred - 200*bound)
            elif (bound < 0) :
                if bound%2 ==1:
                    pred = abs(pred - (bound+1) *200)
                else :
                    pred = pred + (abs(bound)*200)
           
            return move_to(player = '2P',pred = pred)

    # 2. Inform the game process that ml process is ready
    comm.ml_ready()

    # 3. Start an endless loop
    while True:
        # 3.1. Receive the scene information sent from the game process
        scene_info = comm.recv_from_game()
        
        
        
      
        # 3.2. If either of two sides wins the game, do the updating or
        #      resetting stuff and inform the game process when the ml process
        #      is ready.
        feature = []
        feature.append(scene_info["ball"][0])
        feature.append(scene_info["ball"][1])
        feature.append(scene_info["ball_speed"][0])
        feature.append(scene_info["ball_speed"][1])
        feature.append(scene_info["blocker"][0])
        feature.append(scene_info["blocker"][1])
        
        
        feature = np.array(feature)
        feature = feature.reshape((-1,6))
        
        if scene_info["status"] != "GAME_ALIVE":
            # Do some updating or resetting stuff
            ball_served = False

            # 3.2.1 Inform the game process that
            #       the ml process is ready for the next round
            comm.ml_ready()
            continue

        # 3.3 Put the code here to handle the scene information

        # 3.4 Send the instruction for this frame to the game process
        if not ball_served:
            comm.send_to_game({"frame": scene_info["frame"], "command": "SERVE_TO_LEFT"})
            ball_served = True
            stop=False
        else:
           
            # blocker 的x 0~170
            #ball  2p 打在80  1p打在415  2p打blocker在235 1p打在260
            command_1p = ml_loop_for_1P()
            command_2p = ml_loop_for_2P()
            if abs(scene_info["ball_speed"][1])>11 :
                if scene_info["ball"][1]<=435 and scene_info["ball"][1]>=60:
                   ball_x.append(scene_info["ball"][0])
                   ball_y.append(scene_info["ball"][1])
                   ball_speed_x.append(scene_info["ball_speed"][0])
                   ball_speed_y.append(scene_info["ball_speed"][1])
                   brick_x.append(scene_info["blocker"][0])
                   brick_y.append(scene_info["blocker"][1])
                   platform1p.append(scene_info["platform_1P"][0])
                   i=i+1
                   k=k+1
                   if scene_info["ball"][1]==415:
                       for j in range(i):
                          ball_willbe1p_x.append(scene_info["ball"][0])
                       i=0
                   elif scene_info["ball"][1]>415:
                     if len(ball_speed_x)>2:
                       if ball_speed_x[-2]>0: 
                           temp=scene_info["ball"][0]-(scene_info["ball"][1]-415)
                           for j in range(i):
                               ball_willbe1p_x.append(temp)
                       else :
                           temp=scene_info["ball"][0]+(scene_info["ball"][1]-415)
                           for j in range(i):
                               ball_willbe1p_x.append(temp)
                       i=0
                     else:
                       for j in range(i):
                          ball_willbe1p_x.append(scene_info["ball"][0])
                       i=0
                   if scene_info["ball"][1]==80:
                       for l in range(k):
                          ball_willbe2p_x.append(scene_info["ball"][0])
                       k=0
                   elif scene_info["ball"][1]<80:
                     if len(ball_speed_x)>2:
                       if ball_speed_x[-2]>0: 
                           temp=scene_info["ball"][0]-(80-scene_info["ball"][1])
                           for l in range(k):
                               ball_willbe2p_x.append(temp)
                       else :
                           temp=scene_info["ball"][0]+(80-scene_info["ball"][1])
                           for l in range(k):
                               ball_willbe2p_x.append(temp)
                       k=0 
                     else:
                       for l in range(k):
                          ball_willbe2p_x.append(scene_info["ball"][0])
                       k=0  
            if side == "1P":
               # if abs(scene_info["ball_speed"][1])<=13:
                  #      command = ml_loop_for_1P()
                #if abs(scene_info["ball_speed"][1])>13:
               if abs(scene_info["ball_speed"][1])<=11 :
                   command=command_1p
               else:
                   predict = clf_1p.predict(feature)-8
                   if predict<scene_info["platform_1P"][0]+7:
                       command=2
                   elif predict>scene_info["platform_1P"][0]+7:
                       command=1
                   else:
                       command=0
                   
            else:
                command=command_2p
            
            if command == 0:
                comm.send_to_game({"frame": scene_info["frame"], "command": "NONE"})
            elif command == 1:
                comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_RIGHT"})
            else :
                comm.send_to_game({"frame": scene_info["frame"], "command": "MOVE_LEFT"})
            
           
                
                
                