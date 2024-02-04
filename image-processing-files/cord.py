import cv2
import matplotlib.pyplot as plt

def on_click(event):
    # Get the coordinates of the clicked point
    x, y = event.xdata, event.ydata
    
    if x is not None and y is not None:
        # Print the coordinates
        print(f"Coordinates: ({x:.0f}, {y:.0f})")
        
        # Draw a circle with the midpoint as the center
        circle = plt.Circle((x, y), 21, color='r', fill=False)
        ax.add_patch(circle)
        
        # Redraw the image with the circle
        fig.canvas.draw()

# Read the image
img = cv2.imread('im01-RET029OD.jpg')

# Convert the image from BGR to RGB color space (since OpenCV uses BGR and Matplotlib uses RGB)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image using Matplotlib
fig, ax = plt.subplots()
ax.imshow(img_rgb)

# Set the on_click function to be called when the mouse is clicked on the image
cid = fig.canvas.mpl_connect('button_press_event', on_click)

# Show the image window
plt.show()

#Coordinates: (188, 211)






