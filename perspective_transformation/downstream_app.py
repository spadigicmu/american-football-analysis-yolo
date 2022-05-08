import json
import math

import cv2
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from scipy import ndimage
from scipy.spatial import distance


'''

Passing logic

Inputs:	
Player info
Co-ordinates
Team 
Player role

Check if role classification is fine enough to filter who the receiver is
QB is always the thrower

Output
One of
1. If forward pass is best choice Person to pass to
2. If a forward pass is not possible

QB run with the ball
Back/side pass (not sure if these are legal)
Factors
How to identify an eligible receiver
Consider the y-coordinates of the players ie the one with min and max y-coordinates to identify the wide receivers
The distance to the closest opponent
If the distance is lesser than a certain threshold, then it rules out the player from becoming an eligible receiver
How close are the eligible receivers to the target line
How many defenders are blocking the pass trajectory
Even one is a problem
Draw line representing the pass from the QB to the receiver
Identify the perpendicular distance of the opponents from the line
If any one of these distances is below a certain threshold, then eliminate that receiver as the pass is too hard
How many defenders are surrounding an eligible receiver
Where are these defenders located wrt the receiver as in how well are they in positioned to block him from moving forward

'''
def get_bbox_list():
    # Opening JSON file
    f = open('bbox_with_team.json')

    # returns JSON object as
    # a dictionary
    data = json.load(f)

    # plot the bbox with image
    # Create figure and axes
    fig, ax = plt.subplots()
    # Display the image
    img = cv2.imread("../input_images/img1.jpg", cv2.IMREAD_UNCHANGED)
    ax.imshow(img)
    bbox_list = []
    qb = []
    for box in data['predictions']:
        cur_box = box
        curr_bbox = []
        curr_bbox.append(cur_box['x'])
        curr_bbox.append(cur_box['y'])
        curr_bbox.append(cur_box['team'])
        curr_bbox.append(cur_box['class'])
        if (cur_box['class'] == 'QB'):
            qb = box
        bbox_list.append(curr_bbox)
        # Create a Rectangle patch
        rect2 = patches.Rectangle((cur_box['x'] - (cur_box['width'] / 2),
                                   cur_box['y'] - (cur_box['height'] / 2)),
                                  cur_box['width'], cur_box['height'],
                                  linewidth=1, edgecolor='b', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect2)

    #plt.show()
    #cv2.imwrite("../output_images/boxes_demo.png", img)
    bbox_list = np.array(bbox_list) 
    return bbox_list, qb

'''
Get the distance between two players
'''
def get_distance(a, b):
    dst = distance.euclidean(a, b)
    return dst 

'''
Get the team infromation from the json 
'''
def get_teams(players):
    team_a = []
    team_b = []
    for pl in players:
        if pl[2] == '0':
            team_a.append(pl[:2])
        else:
            team_b.append(pl[:2])
    return team_a, team_b 

'''
Plot the lines for possible players who can receive pass from QB
'''
def plot_projected_passes(img, qb, proj_players, plot_name="projected_passes.png"):
    colors = [(255, 255, 255), (255,255,0),(128,0,128), (255,0,0)]
    for i,player in enumerate(proj_players):      
        color = colors[i] 
        cv2.line(img, (int(float(qb[0])), int(float(qb[1]))), (int(float(player[0])), int(float((player[1])))), color, 1)

    cv2.imwrite("../output_images/" + plot_name, img)
    return img 

'''
Get the possible players who can receive pass from QB
'''
def get_passable_players(qb, players, img):
    a = (float(qb['x']), float(qb['y']))
    qb_dist = []

    for player in players:
        b = (float(player[0]), float(player[1]))
        qb_dist.append(get_distance(a, b))

    norm_dist = [float(i)/max(qb_dist) for i in qb_dist]

    proj_players = []
    for i,dist in enumerate(norm_dist):
        if dist > 0.5:
            proj_players.append(players[i])

    img = plot_projected_passes(img, a, proj_players)
    return proj_players, img 

'''
Plot the lines for possible tacklers 
'''
def plot_top_tacklers(img, proj_players, top_tacklers, top_tacklers_dist):
    colors = [(255, 255, 255), (255,255,0),(128,0,128), (255,0,0)]
    avg_tacklability = []
    for i,player in enumerate(proj_players):
        a = (int(float(player[0])), int(float((player[1]))))
        color = colors[i]
        
        for j in range(3):
            top = top_tacklers[i][j]
            b = (int(float(top[0])), int(float((top[1]))))
            #print(a,b)
            cv2.line(img, a, b, color, 1)
        avg_tacklability.append(sum(top_tacklers_dist[i])/len(top_tacklers_dist[i]))
    cv2.imwrite("../output_images/tacklers.png", img)
    return img, avg_tacklability

def argsort(seq, reversed = False):
    return sorted(range(len(seq)), key=seq.__getitem__, reverse=reversed)

'''
Get the prospects of players

Algorithm: 
1. Find the wide players who can recieve the pass 
2. For each identified player, find the top 3 closest tacklers 
3. Rank the passers based on the distances of top 3 tacklers  
4. Return the one with least "tackle potential"
'''
def get_passer_prospects(team_a, team_b, qb, img):
    orig_img = img
    proj_players, img = get_passable_players(qb, team_a, img)
    top_tacklers = []
    top_tacklers_positions = []
    for player in proj_players:
        a = (float(player[0]), float(player[1]))
        tacklers = []
            
        # get the distance of all tacklers
        for opponent in team_b:
            b = (float(opponent[0]), float(opponent[1]))
            tacklers.append(get_distance(a, b)) 
        #norm_dist = [float(i)/sum(tacklers) for i in tacklers]
        player_args = argsort(tacklers)
        norm_dist = sorted(tacklers, reverse=False)
        top_tacklers_temp = []
        #print("player args", player_args)
        for pos in player_args[:3]:
            top_tacklers_temp.append(team_b[pos])
        #print(top_tacklers_temp)
        # get top 3 tacklers 
        norm_dist = norm_dist[:3]
        top_tacklers.append(norm_dist)
        top_tacklers_positions.append(top_tacklers_temp)
    #print(top_tacklers)
    #print(top_tacklers_positions)
    img, tacklability = plot_top_tacklers(img, proj_players, top_tacklers_positions, top_tacklers)
    print("tacklability", tacklability)
    best_pass_position = argsort(tacklability, True)[0]
    a = (float(qb['x']), float(qb['y']))
    orig_img = cv2.imread("../input_images/img1.jpg", cv2.IMREAD_UNCHANGED)

    plot_projected_passes(orig_img, a, [proj_players[best_pass_position]], "best_pass.png")


image = cv2.imread("../input_images/img1.jpg", cv2.IMREAD_UNCHANGED)
points, qb = get_bbox_list()
team_a, team_b = get_teams(points) 
print("team a", team_a)
print("team b", team_b)
get_passable_players(qb, team_a, image)

get_passer_prospects(team_a, team_b, qb, image)