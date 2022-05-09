from PIL import Image  
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
import numpy as np
import json
import math
import cv2
import numpy as np
from collections import Counter

img_path = "../input_images/img1.jpg"
label_path = "bbox.json"


# load the image 
img = Image.open(img_path).convert("RGB") 

trans1 = transforms.ToTensor()
img_tensor = trans1(img)
print(img_tensor.size()) 

_, height, width = img_tensor.size()

# load the label
# x, y, width, height
def get_bbox_list():
    # Opening JSON file
    f = open('bbox.json')

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
        curr_bbox.append(int(float(cur_box['x'])))
        curr_bbox.append(int(float(cur_box['y'])))
        curr_bbox.append(int(float(cur_box['width'])))
        curr_bbox.append(int(float(cur_box['height'])))
        #curr_bbox.append(cur_box['class'])
        #if (cur_box['class'] == 'QB'):
        #    qb = box
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
    #bbox_list = np.array(bbox_list) 
    #print(bbox_list.shape)
    return bbox_list, qb

b_boxes, _ = get_bbox_list()
# b_boxes = []
# with open(label_path) as labels:
#   for line in labels.readlines():
#     box = line.strip().split(' ')[1:]
#     box = [float(b) for b in box] 
#     box[0] =  box[0] * width
#     box[1] =  box[1] * height 
#     box[2] =  box[2] * width
#     box[3] =  box[3] * height  
    
#     print(box)
#     b_boxes.append(box)


# plot the bbox with image
# Create figure and axes
#fig, ax = plt.subplots()
# Display the image
#ax.imshow(img)
#for box in b_boxes:
  #cur_box = box
  #print(cur_box)
  # Create a Rectangle patch
  #rect2 = patches.Rectangle((cur_box[0]-(cur_box[2]/2), cur_box[1]-(cur_box[3]/2)), cur_box[2], cur_box[3], linewidth=1, edgecolor='b', facecolor='none')
  # Add the patch to the Axes
  #ax.add_patch(rect2)

#plt.show()

pallete = {'b': (0, 0, 128),
          'g': (0, 128, 0),
          'r': (255, 0, 0),
          'c': (0, 192, 192),
          'm': (192, 0, 192),
          'y': (192, 192, 0),
          'k': (0, 0, 0),
          'w': (255, 255, 255)}

N_CLUSTERS = 5

# Find the closest color to the detected one based on the predefined palette
def closest_color(list_of_colors, color):
    colors = np.array(list_of_colors)
    color = np.array(color)
    distances = np.sqrt(np.sum((colors-color)**2,axis=1))
    index_of_shortest = np.where(distances==np.amin(distances))
    shortest_distance = colors[index_of_shortest]

    return shortest_distance


def get_player_color(player_index ):
  # get one players image 

  cur_box = b_boxes[player_index]
  
  startX = int(cur_box[0]-(cur_box[2]/2))
  startY = int(cur_box[1]-(cur_box[3]/2))
  endX = int(cur_box[0]+(cur_box[2]/2))
  endY = int(cur_box[1]+(cur_box[3]/2))

  player_image = img_tensor[:, startY:endY, startX:endX] 

  #plt.imshow(player_image.permute(1,2,0))
  #plt.show()
  # KMeans for clustering 
  n_colors = N_CLUSTERS
  kmeans = KMeans(n_clusters = n_colors, n_init = 20)

  #print("player before: ", player_image)
  player_image = player_image.reshape(3, -1).permute(1, 0)
  #print("player after: ", player_image)

  k_clusters = kmeans.fit(player_image * 255)

  # K-MEANS result
  centroid = kmeans.cluster_centers_
  labels = kmeans.labels_


  labels = list(labels)
  percent = []

  for i in range(len(centroid)):
      j = labels.count(i)
      j = j / len(labels)
      percent.append(j)

  #print("percent: ", percent)
  sorted_indices = np.argsort(percent)
  # Get the majority color 
  assigned_cluster = sorted_indices[-1] 
  #print("assigned cluster index", assigned_cluster) 
  detected_color = centroid[assigned_cluster]

  list_of_colors = list(pallete.values())
  assigned_color = closest_color(list_of_colors, detected_color)[0]
  assigned_color = (int(assigned_color[0]), int(assigned_color[1]), int(assigned_color[2]))
  # if assigned color is green then get the next color. 
  color_index = -2
  print("original color", assigned_color, assigned_color == (0, 128, 0))
  while (assigned_color == (192, 192, 0) and color_index <= 0):
    assigned_cluster = sorted_indices[color_index]
    color_index -= 1 
    print("assigned cluster index", assigned_cluster) 
    detected_color = centroid[assigned_cluster]

    list_of_colors = list(pallete.values())
    assigned_color = closest_color(list_of_colors, detected_color)[0]
    assigned_color = (int(assigned_color[0]), int(assigned_color[1]), int(assigned_color[2]))
  print("assigned_color: ", assigned_color)


  for k, v in pallete.items():
    if (v == assigned_color):
      color = k
      break

  rect2 = patches.Rectangle((cur_box[0]-(cur_box[2]/2), cur_box[1]-(cur_box[3]/2)), cur_box[2], cur_box[3], linewidth=3, edgecolor=color, facecolor='none')
  # Add the patch to the Axes
  ax.add_patch(rect2)
  if assigned_color == (0, 0, 0):
      assigned_color = (128, 128, 128)

  #print(centroid)


# fig, ax = plt.subplots(figsize=(15,15))
# # Display the image
# ax.imshow(img)
# for i in range(14):
#   get_player_color(i)

# plt.show()


pallete = {
  'Black': (0,0,0),
  #'White': (255,255,255),
  'Red': (255,0,0),
  'Lime': (0,255,0),
  'Blue': (0,0,255),
  'Yellow': (255,255,0),
  'Cyan': (0,255,255),
  'Magenta': (255,0,255),
  #'Silver': (192,192,192),
  #'Gray': (128,128,128),
  'Maroon': (128,0,0),
  #'Olive': (128,128,0),
  'Green': (0,128,0),
  'Purple': (128,0,128),
  'Teal': (0,128,128),
  'Navy': (0,0,128)
  }

N_CLUSTERS = 1

def get_player_color(player_index):
  # get one players image 

  cur_box = b_boxes[player_index]
  startX = int(cur_box[0]-(cur_box[2]/2))
  startY = int(cur_box[1]-(cur_box[3]/2))
  endX = int(cur_box[0]+(cur_box[2]/2))
  endY = int(cur_box[1]+(cur_box[3]/2))

  player_image = img_tensor[:, startY:endY, startX:endX] 
  # KMeans for clustering 
  n_colors = N_CLUSTERS
  kmeans = KMeans(n_clusters = n_colors, n_init = 20)

  #print("player before: ", player_image)
  player_image = player_image.reshape(3, -1).permute(1, 0)
  #print("player after: ", player_image)

  k_clusters = kmeans.fit(player_image * 255)

  # K-MEANS result
  centroid = kmeans.cluster_centers_
  labels = kmeans.labels_


  labels = list(labels)
  percent = []

  for i in range(len(centroid)):
      j = labels.count(i)
      j = j / len(labels)
      percent.append(j)

  #print("percent: ", percent)
  sorted_indices = np.argsort(percent)
  # Get the majority color 
  assigned_cluster = sorted_indices[-1] 
  #print("assigned cluster index", assigned_cluster) 
  detected_color = centroid[assigned_cluster]

  list_of_colors = list(pallete.values())
  assigned_color = closest_color(list_of_colors, detected_color)[0]
  assigned_color = (int(assigned_color[0]), int(assigned_color[1]), int(assigned_color[2]))
  # if assigned color is green then get the next color. 
  color_index = -2
  print("original color", assigned_color, assigned_color == (0, 128, 0))
  while (assigned_color == (0, 128, 0) and color_index <= 0):
    assigned_cluster = sorted_indices[color_index]
    color_index -= 1 
    print("assigned cluster index", assigned_cluster) 
    detected_color = centroid[assigned_cluster]

    list_of_colors = list(pallete.values())
    assigned_color = closest_color(list_of_colors, detected_color)[0]
    assigned_color = (int(assigned_color[0]), int(assigned_color[1]), int(assigned_color[2]))
  print("assigned_color: ", assigned_color)

  if assigned_color == ((0,128,128)):
    assigned_color = (0,0,0)

  for k, v in pallete.items():
    if (v == assigned_color):
      color = k
      break

  rect2 = patches.Rectangle((cur_box[0]-(cur_box[2]/2), cur_box[1]-(cur_box[3]/2)), cur_box[2], cur_box[3], linewidth=3, edgecolor=color, facecolor='none')
  # Add the patch to the Axes
  ax.add_patch(rect2)
  if assigned_color == (0, 0, 0):
      assigned_color = (128, 128, 128)
  
  return assigned_color

def write_to_json(team_info, filepath="bbox_team_updated.json"):

    og_json = json.load(open('bbox.json'))
    # Writing to sample.json
    for entry in range(len(og_json['predictions'])):
        og_json['predictions'][entry]['team'] = team_info[entry]
        #og_json['predictions'][entry]['y'] = points_info[entry][1]
        if og_json['predictions'][entry]['class'] == 'QB':
            team_0 =  team_info[entry]
    
    for entry in range(len(og_json['predictions'])):
        if og_json['predictions'][entry]['team'] == team_0:
            og_json['predictions'][entry]['team'] = 0
        else:
            og_json['predictions'][entry]['team'] = 1

    json_object = json.dumps(og_json, indent=4)
    with open(filepath, "w") as outfile:
        outfile.write(json_object)


fig, ax = plt.subplots(figsize=(10,15))
# Display the image
ax.imshow(img)
team_assignments = []
for i in range(len(b_boxes)):
  team = get_player_color(i)
  team_assignments.append(team)

counts = Counter(team_assignments)
# get the two majority colors 
two_major_colors = [] 
two_major_colors.append(list(counts.keys())[0])
two_major_colors.append(list(counts.keys())[1])
for i,team in enumerate(team_assignments):
    color = closest_color(two_major_colors, team).tolist()[0]
    color = (color[0], color[1], color[2])
    if color not in two_major_colors:
        team_assignments[i] = color 

print(team_assignments)
write_to_json(team_assignments)

