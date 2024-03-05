#!/usr/bin/env python
# coding: utf-8

# In[1]:


#interesting article https://en.wikipedia.org/wiki/Kernel_(image_processing)
import cv2
#cv2 reads in channel as BGR
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#matplotlib reads in channels as RGB
import re



def load_images(image_files, base_directory):
    images = []
    reference_size = None
    for file in image_files:
        path = os.path.join(base_directory, file)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is not None:
            if reference_size is None:
                reference_size = image.shape[1], image.shape[0]
            resized_image = cv2.resize(image, reference_size, interpolation=cv2.INTER_AREA)
            images.append(resized_image)
        else:
            print(f"Failed to load image: {file}")
    return images, reference_size

def preprocess_images(images):
    processed_images = []
    for image in images:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_images.append(gray_image)
    return processed_images

def compute_differences(images):
    reference_image = images[0]
    differences = []
    for image in images[1:]:
        difference = cv2.absdiff(reference_image, image)
        differences.append(difference)
    return differences

def analyze_changes(differences):
    change_analysis = []
    change_stats = []
    for diff in differences:
        ret, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area = sum(cv2.contourArea(contour) for contour in contours)
        change_analysis.append(contours)
        change_stats.append({'count': len(contours), 'area': total_area})
    return change_analysis, change_stats

# Define your image files and base directory
image_files = [
        "Waterbody_052023.png",
    "Waterbody_032013.png",
    "Waterbody_042011.png",
    "Waterbody_052011.png",
    "Waterbody_052013.png",
    "Waterbody_042015.png",
    "Waterbody_032019.png",
    "Waterbody_092020.png",
    "Waterbody_052021.png",
    "Waterbody_032022.png",
    "Waterbody_062022.png",
    "Waterbody_032022.png",
]
base_directory = "/Users/breisdas/Mastermind/Velachery_waterbody/Images/"
reference_image_path = os.path.join(base_directory, image_files[0])
reference_image = cv2.imread(reference_image_path)

def extract_date(filepath):
    filename = os.path.basename(filepath)
    match = re.search(r'Waterbody_(\d{2})(\d{4})\.png', filename)
    if match:
        date_str = match.group(1) + match.group(2)
        return datetime.strptime(date_str, '%m%Y')
    else:
        raise ValueError(f"Filename {filename} does not contain a valid date in MMYYYY format.")
        
def find_waterbody_contours(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define range for blue colors typically representing water
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([150, 255, 255])
    # Create a binary mask where blue colors are white and the rest are black
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_waterbody_contours(image, contours):
    # Draw contours on the image
    cv2.drawContours(image, contours, -1, (255, 0, 0), 2)
    return image


# Load and preprocess the images
images, reference_size = load_images(image_files, base_directory)
if not images:
    print("No images could be loaded, exiting the analysis.")
else:
    processed_images = preprocess_images(images)
    differences = compute_differences(processed_images)
    changes, stats = analyze_changes(differences)

    # Find the image with the biggest change
    max_change_idx = max(range(len(stats)), key=lambda i: stats[i]['area'])
    biggest_change_image = image_files[max_change_idx + 1]

    # Display all images with changes
    for idx, change in enumerate(changes):
        plt.figure(figsize=(5,5))
        if change:
            for contour in change:
                # Green contours represent the detected changes
                cv2.drawContours(images[idx+1], contour, -1, (0,255,0), 3)
            plt.imshow(cv2.cvtColor(images[idx+1], cv2.COLOR_BGR2RGB))
            plt.title(f'Change Detected in Image {image_files[idx+1]}')
        else:
            plt.imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
            plt.title('No Change Detected')
        plt.axis('off')
        plt.show()

    # Plot the graph of changes over the years
    dates = [extract_date(file) for file in image_files[1:]]
    changes_by_date = {date: stat['area'] for date, stat in zip(dates, stats)}
    sorted_dates = sorted(changes_by_date)
    sorted_changes = [changes_by_date[date] for date in sorted_dates]

    plt.figure(figsize=(10, 5))
    plt.plot([date.strftime('%Y') for date in sorted_dates], sorted_changes, marker='o')
    plt.xlabel('Year')
    plt.ylabel('Total Area of Change')
    plt.title('Lake Area Changes Over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    ##Display the reference image
    
if reference_image is not None:
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
    plt.title('Reference Image')
    plt.axis('off')  # Hide the axis
    plt.show()
else:
    print("Reference image is not available.")    

    
# Now, when you want to display the reference image with the biggest change
reference_image_with_biggest_change_path = os.path.join(base_directory, biggest_change_image)
reference_image_with_biggest_change = cv2.imread(reference_image_with_biggest_change_path)
if reference_image_with_biggest_change is None:
    print(f"Failed to load the image with the biggest change from file: {reference_image_with_biggest_change_path}")
else:
    reference_image_with_biggest_change_resized = cv2.resize(reference_image_with_biggest_change, (reference_image.shape[1], reference_image.shape[0]), interpolation=cv2.INTER_AREA)
    contours = find_waterbody_contours(reference_image_with_biggest_change_resized)
    reference_image_with_contours = draw_waterbody_contours(reference_image_with_biggest_change_resized.copy(), contours)  # Make a copy to draw on
    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(reference_image_with_contours, cv2.COLOR_BGR2RGB))
    plt.title(f'Image that represents the biggest change in comparisson with the reference image: {biggest_change_image}')
    plt.axis('off')
    plt.show()
    

# Display the reference image with the biggest change
    reference_image_with_biggest_change = cv2.imread(os.path.join(base_directory, biggest_change_image))
    reference_image_with_biggest_change = cv2.resize(reference_image_with_biggest_change, reference_size, interpolation=cv2.INTER_AREA)
    reference_contours = find_waterbody_contours(reference_image_with_biggest_change)
    reference_image_with_contours = draw_waterbody_contours(reference_image, reference_contours)
    plt.figure(figsize=(10,10))
    plt.imshow(reference_image_with_contours)
    plt.title(f'Reference image in comparisson with the image with the biggest change: {biggest_change_image}')
    plt.axis('off')
    plt.show()
    
    
    # Print summary of changes
    print("\nSummary of Detected Changes:")
    for file, stat in zip(image_files[1:], stats):
        print(f"Image {file}: {stat['count']} regions changed with a total area of {stat['area']:.2f} pixels")
        
# Assuming images is a list of images loaded earlier and reference_image is the first one in this list
reference_image = images[0]  # This line ensures reference_image is defined

#Step 1: Function to Find and Display the Largest Change
def display_largest_change_image(images, changes):
    largest_area = 0
    largest_contour = None  # Directly store the largest contour
    largest_image_index = None

    for idx, contours in enumerate(changes):
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > largest_area:
                largest_area = area
                largest_contour = contour  # Update largest contour
                largest_image_index = idx + 1  # +1 because 'changes' excludes the reference image

    if largest_image_index is not None and largest_contour is not None:
        # Draw the largest contour on the image
        cv2.drawContours(images[largest_image_index], [largest_contour], -1, (0, 255, 0), 3)
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(images[largest_image_index], cv2.COLOR_BGR2RGB))
        plt.title(f"Image with the Largest Change: {image_files[largest_image_index]}")
        plt.axis('off')
        plt.show()
    else:
        print("No significant changes detected.")

display_largest_change_image(images, changes)


#Step 2: Rudimentary Summary of Changes
def print_summary_of_changes(stats):
    total_changed_area = sum(stat['area'] for stat in stats)
    total_regions_changed = sum(stat['count'] for stat in stats)
    print(f"Total changed area across all images: {total_changed_area:.2f} pixels")
    print(f"Total regions of change detected: {total_regions_changed}")

print_summary_of_changes(stats)

def display_largest_waterbody_change(images, changes):
    largest_change_area = 0
    largest_change_image_index = None
    largest_change_contour = None

    # Iterate through each image's changes to find the largest change in waterbody
    for idx, contour_list in enumerate(changes):
        for contour in contour_list:
            area = cv2.contourArea(contour)
            if area > largest_change_area:
                largest_change_area = area
                largest_change_image_index = idx + 1  # Adding 1 to account for reference image
                largest_change_contour = contour

    # Display the image with the largest change in waterbody area
    if largest_change_image_index is not None:
        img_with_largest_change = images[largest_change_image_index].copy()
        cv2.drawContours(img_with_largest_change, [largest_change_contour], -1, (0, 255, 0), 3)
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img_with_largest_change, cv2.COLOR_BGR2RGB))
        plt.title(f"Image with Largest Waterbody Change: {image_files[largest_change_image_index]}")
        plt.axis('off')
        plt.show()
    else:
        print("No significant waterbody changes detected.")

# Call the function after analyzing the changes
display_largest_waterbody_change(images, changes)

def analyze_encroachment(images, changes):
    waterbody_areas = []
    for contours in changes:
        # Assuming the largest contour in each image is the waterbody
        largest_contour = max(contours, key=cv2.contourArea) if contours else None
        area = cv2.contourArea(largest_contour) if largest_contour is not None else 0
        waterbody_areas.append(area)

    # Assuming the first image is the reference, calculate the change in area
    # Negative values indicate a reduction in waterbody area (potential encroachment)
    area_changes = [waterbody_areas[0] - area for area in waterbody_areas]
    
    # Find the image with the largest reduction in waterbody area
    min_change = min(area_changes)
    if min_change < 0:  # Ensuring that there's a reduction
        encroachment_image_index = area_changes.index(min_change) + 1  # +1 to adjust index
        print(f"Image with the biggest encroachment: {image_files[encroachment_image_index]}")
        
        # Display the image with the biggest encroachment
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(images[encroachment_image_index], cv2.COLOR_BGR2RGB))
        plt.title(f"Biggest Encroachment: {image_files[encroachment_image_index]}")
        plt.axis('off')
        plt.show()
    else:
        print("No significant encroachment detected.")

# Call this function after analyzing the changes
analyze_encroachment(images, changes)



# In[2]:


def display_largest_expansion_image(images, changes, stats):
    # Initialize variables to track the largest expansion
    largest_expansion_area = 0
    image_with_largest_expansion_index = None

    # Calculate the area of the waterbody in the reference image for comparison
    reference_area = sum(cv2.contourArea(contour) for contour in changes[0])

    # Loop through the stats to find the largest expansion
    for idx, stat in enumerate(stats[1:], start=1): # Skip the reference image
        current_area = stat['area']
        expansion = current_area - reference_area
        if expansion > largest_expansion_area:
            largest_expansion_area = expansion
            image_with_largest_expansion_index = idx

    if image_with_largest_expansion_index is not None:
        # Draw all contours on the image with the largest expansion
        for contour in changes[image_with_largest_expansion_index-1]: # -1 because 'changes' includes the reference image at index 0
            cv2.drawContours(images[image_with_largest_expansion_index], [contour], -1, (0, 255, 0), 3)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(images[image_with_largest_expansion_index], cv2.COLOR_BGR2RGB))
        plt.title(f"Image with the Largest Lake Expansion: {image_files[image_with_largest_expansion_index]}")
        plt.axis('off')
        plt.show()
    else:
        print("No expansion detected or data insufficient for comparison.")

# Call the function to display the image with the largest lake expansion
display_largest_expansion_image(images, changes, stats)


# In[3]:


def find_largest_expansion(changes, stats):
    # Assuming 'stats' contains the area of waterbody for each image
    # Expansion is identified by an increase in waterbody area compared to the reference
    max_expansion_area = 0
    expansion_index = None
    expansion_contour = None

    for i, stat in enumerate(stats[1:], 1):  # Skip reference image at index 0
        if stat['area'] > max_expansion_area:
            max_expansion_area = stat['area']
            expansion_index = i
            expansion_contour = max(changes[i-1], key=cv2.contourArea)  # -1 because 'changes' includes reference at index 0

    return expansion_index, expansion_contour

def find_largest_encroachment(changes, stats):
    # Assuming 'stats' contains the area of waterbody for each image
    # Encroachment is identified by a decrease in waterbody area compared to the reference
    min_encroachment_area = float('inf')
    encroachment_index = None
    encroachment_contour = None

    for i, stat in enumerate(stats[1:], 1):  # Skip reference image at index 0
        if stat['area'] < min_encroachment_area:
            min_encroachment_area = stat['area']
            encroachment_index = i
            encroachment_contour = max(changes[i-1], key=cv2.contourArea)  # -1 because 'changes' includes reference at index 0

    return encroachment_index, encroachment_contour

def overlay_encroachment_on_expansion(images, changes, stats):
    expansion_index, expansion_contour = find_largest_expansion(changes, stats)
    encroachment_index, encroachment_contour = find_largest_encroachment(changes, stats)

    if expansion_index is not None and encroachment_index is not None:
        image_with_overlay = images[expansion_index].copy()
        cv2.drawContours(image_with_overlay, [expansion_contour], -1, (0, 255, 0), 3)  # Draw expansion in green
        cv2.drawContours(image_with_overlay, [encroachment_contour], -1, (0, 0, 255), 3)  # Draw encroachment in red

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(image_with_overlay, cv2.COLOR_BGR2RGB))
        plt.title("Overlay of Biggest Encroachment on Largest Expansion")
        plt.axis('off')
        plt.show()
    else:
        print("Could not find the largest expansion or biggest encroachment.")

# Assuming you have already defined 'images', 'changes', and 'stats'
# Call the function with actual parameters
overlay_encroachment_on_expansion(images, changes, stats)


def display_largest_expansion_image(images, changes, stats):
    # Initialize variables to track the largest expansion
    largest_expansion_area = 0
    image_with_largest_expansion_index = None

    # Calculate the area of the waterbody in the reference image for comparison
    reference_area = sum(cv2.contourArea(contour) for contour in changes[0])

    # Loop through the stats to find the largest expansion
    for idx, stat in enumerate(stats[1:], start=1): # Skip the reference image
        current_area = stat['area']
        expansion = current_area - reference_area
        if expansion > largest_expansion_area:
            largest_expansion_area = expansion
            image_with_largest_expansion_index = idx

    if image_with_largest_expansion_index is not None:
        # Draw all contours on the image with the largest expansion
        for contour in changes[image_with_largest_expansion_index-1]: # -1 because 'changes' includes the reference image at index 0
            cv2.drawContours(images[image_with_largest_expansion_index], [contour], -1, (0, 255, 0), 3)
        
        plt.figure(figsize=(5, 5))
        plt.imshow(cv2.cvtColor(images[image_with_largest_expansion_index], cv2.COLOR_BGR2RGB))
        plt.title(f"Image with the Largest Lake Expansion: {image_files[image_with_largest_expansion_index]}")
        plt.axis('off')
        plt.show()
    else:
        print("No expansion detected or data insufficient for comparison.")

# Call the function to display the image with the largest lake expansion
display_largest_expansion_image(images, changes, stats)


def analyze_encroachment(images, changes):
    waterbody_areas = []
    for contours in changes:
        # Assuming the largest contour in each image is the waterbody
        largest_contour = max(contours, key=cv2.contourArea) if contours else None
        area = cv2.contourArea(largest_contour) if largest_contour is not None else 0
        waterbody_areas.append(area)

    # Assuming the first image is the reference, calculate the change in area
    # Negative values indicate a reduction in waterbody area (potential encroachment)
    area_changes = [waterbody_areas[0] - area for area in waterbody_areas]
    
    # Find the image with the largest reduction in waterbody area
    min_change = min(area_changes)
    if min_change < 0:  # Ensuring that there's a reduction
        encroachment_image_index = area_changes.index(min_change) + 1  # +1 to adjust index
        print(f"Image with the biggest encroachment: {image_files[encroachment_image_index]}")
        
        # Display the image with the biggest encroachment
        plt.figure(figsize=(5, 5))
        plt.imshow(cv2.cvtColor(images[encroachment_image_index], cv2.COLOR_BGR2RGB))
        plt.title(f"Biggest Encroachment: {image_files[encroachment_image_index]}")
        plt.axis('off')
        plt.show()
    else:
        print("No significant encroachment detected.")

# Call this function after analyzing the changes
analyze_encroachment(images, changes)



# In[4]:


import pytesseract

# Function to perform OCR on an image and return detected text and bounding boxes
def perform_ocr(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Perform thresholding to preprocess the image
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Perform OCR using pytesseract
    text_data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
    return text_data

# Function to filter out non-empty text regions and return their bounding boxes
def get_text_regions(text_data):
    text_regions = []
    for i in range(len(text_data['text'])):
        if text_data['text'][i] != '':
            x, y, w, h = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i]
            text_regions.append((x, y, w, h))
    return text_regions

# Function to check if a point (x, y) lies inside any contour
def is_point_inside_contour(x, y, contour):
    return cv2.pointPolygonTest(contour, (x, y), False) >= 0

# Function to determine if a region is covered by water based on the intersection of text regions and waterbody contours
def determine_water_coverage(image, text_regions, waterbody_contours):
    water_coverage = False
    for region in text_regions:
        # Check if the center of the region lies inside any waterbody contour
        x, y, w, h = region
        region_center_x = x + w // 2
        region_center_y = y + h // 2
        for contour in waterbody_contours:
            if is_point_inside_contour(region_center_x, region_center_y, contour):
                water_coverage = True
                break
        if water_coverage:
            break
    return water_coverage

# Function to display the image with water coverage information
def display_image_with_water_coverage(image, water_coverage):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if water_coverage:
        plt.title("Area Covered by Water")
    else:
        plt.title("Area Not Covered by Water")
    plt.axis('off')
    plt.show()

# Assuming 'images' and 'changes' are available from previous processing
# Assuming 'reference_image' and 'biggest_change_image' are available

# Perform OCR on the image with the biggest change
biggest_change_image_with_water_text = images[max_change_idx + 1].copy()  # Make a copy to preserve original
text_data = perform_ocr(biggest_change_image_with_water_text)

# Extract bounding boxes of text regions
text_regions = get_text_regions(text_data)

# Find contours in the image with the biggest change
waterbody_contours_biggest_change = find_waterbody_contours(biggest_change_image_with_water_text)

# Determine if the detected text regions intersect with waterbody contours
water_coverage = determine_water_coverage(biggest_change_image_with_water_text, text_regions, waterbody_contours_biggest_change)

# Display the image with water coverage information
display_image_with_water_coverage(biggest_change_image_with_water_text, water_coverage)


# In[5]:


import pytesseract

# Function to perform OCR on an image and return detected text and bounding boxes
def perform_ocr(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Perform thresholding to preprocess the image
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Perform OCR using pytesseract
    text_data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
    return text_data

# Function to filter out non-empty text regions and return their bounding boxes
def get_text_regions(text_data):
    text_regions = []
    for i in range(len(text_data['text'])):
        if text_data['text'][i] != '':
            x, y, w, h = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i]
            text_regions.append((x, y, w, h))
    return text_regions

# Function to check if a point (x, y) lies inside any contour
def is_point_inside_contour(x, y, contour):
    return cv2.pointPolygonTest(contour, (x, y), False) >= 0

# Function to determine if a region is covered by water based on the intersection of text regions and waterbody contours
def determine_water_coverage(image, text_regions, waterbody_contours):
    water_text_regions = []
    non_water_text_regions = []
    for region in text_regions:
        # Check if the center of the region lies inside any waterbody contour
        x, y, w, h = region
        region_center_x = x + w // 2
        region_center_y = y + h // 2
        water_coverage = False
        for contour in waterbody_contours:
            if is_point_inside_contour(region_center_x, region_center_y, contour):
                water_coverage = True
                break
        if water_coverage:
            water_text_regions.append(region)
        else:
            non_water_text_regions.append(region)
    return water_text_regions, non_water_text_regions

# Function to display the image with water coverage information
def display_image_with_water_coverage(image, water_text_regions, non_water_text_regions):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for region in water_text_regions:
        x, y, w, h = region
        plt.Rectangle((x, y), w, h, fill=None, edgecolor='blue', linewidth=2)
    for region in non_water_text_regions:
        x, y, w, h = region
        plt.Rectangle((x, y), w, h, fill=None, edgecolor='red', linewidth=2)
    plt.title("Text Regions Covered by Water (Blue) and Not Covered by Water (Red)")
    plt.axis('off')
    plt.show()

# Assuming 'images' and 'changes' are available from previous processing
# Assuming 'reference_image' and 'biggest_change_image' are available

# Perform OCR on the image with the biggest change
biggest_change_image_with_water_text = images[max_change_idx + 1].copy()  # Make a copy to preserve original
text_data = perform_ocr(biggest_change_image_with_water_text)

# Extract bounding boxes of text regions
text_regions = get_text_regions(text_data)

# Find contours in the image with the biggest change
waterbody_contours_biggest_change = find_waterbody_contours(biggest_change_image_with_water_text)

# Determine if the detected text regions intersect with waterbody contours
water_text_regions, non_water_text_regions = determine_water_coverage(biggest_change_image_with_water_text, text_regions, waterbody_contours_biggest_change)

# Display the image with water coverage information
display_image_with_water_coverage(biggest_change_image_with_water_text, water_text_regions, non_water_text_regions)


# In[6]:


import pytesseract
import matplotlib.pyplot as plt
import cv2

# Function to perform OCR on an image and return detected text and bounding boxes
def perform_ocr(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Perform thresholding to preprocess the image
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Perform OCR using pytesseract
    text_data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
    return text_data

# Function to filter out non-empty text regions and return their bounding boxes
def get_text_regions(text_data):
    text_regions = []
    for i in range(len(text_data['text'])):
        if text_data['text'][i] != '':
            x, y, w, h = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i]
            text_regions.append((x, y, w, h))
    return text_regions

# Function to check if a point (x, y) lies inside any contour
def is_point_inside_contour(x, y, contour):
    return cv2.pointPolygonTest(contour, (x, y), False) >= 0

# Function to determine if a region is covered by water based on the intersection of text regions and waterbody contours
def determine_water_coverage(image, text_regions, waterbody_contours):
    water_text_regions = []
    non_water_text_regions = []
    for region in text_regions:
        # Check if the center of the region lies inside any waterbody contour
        x, y, w, h = region
        region_center_x = x + w // 2
        region_center_y = y + h // 2
        water_coverage = False
        for contour in waterbody_contours:
            if is_point_inside_contour(region_center_x, region_center_y, contour):
                water_coverage = True
                break
        if water_coverage:
            water_text_regions.append(region)
        else:
            non_water_text_regions.append(region)
    return water_text_regions, non_water_text_regions

# Function to display the image with water coverage information
def display_image_with_water_coverage(image, water_text_regions, non_water_text_regions):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for region in water_text_regions:
        x, y, w, h = region
        plt.Rectangle((x, y), w, h, fill=None, edgecolor='blue', linewidth=2)  # Use plt.Rectangle instead of plt.Rectangle
    for region in non_water_text_regions:
        x, y, w, h = region
        plt.Rectangle((x, y), w, h, fill=None, edgecolor='red', linewidth=2)  # Use plt.Rectangle instead of plt.Rectangle
    plt.title("Text Regions Covered by Water (Blue) and Not Covered by Water (Red)")
    plt.axis('off')
    plt.show()

# Assuming 'images' and 'changes' are available from previous processing
# Assuming 'reference_image' and 'biggest_change_image' are available

# Perform OCR on the image with the biggest change
biggest_change_image_with_water_text = images[max_change_idx + 1].copy()  # Make a copy to preserve original
text_data = perform_ocr(biggest_change_image_with_water_text)

# Extract bounding boxes of text regions
text_regions = get_text_regions(text_data)

# Find contours in the image with the biggest change
waterbody_contours_biggest_change = find_waterbody_contours(biggest_change_image_with_water_text)

# Determine if the detected text regions intersect with waterbody contours
water_text_regions, non_water_text_regions = determine_water_coverage(biggest_change_image_with_water_text, text_regions, waterbody_contours_biggest_change)

# Display the image with water coverage information
display_image_with_water_coverage(biggest_change_image_with_water_text, water_text_regions, non_water_text_regions)


# In[7]:


import pytesseract
import matplotlib.pyplot as plt
import cv2

# Function to perform OCR on an image and return detected text and bounding boxes
def perform_ocr(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Perform thresholding to preprocess the image
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Perform OCR using pytesseract
    text_data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
    return text_data

# Function to filter out non-empty text regions and return their bounding boxes
def get_text_regions(text_data):
    text_regions = []
    for i in range(len(text_data['text'])):
        if text_data['text'][i] != '':
            x, y, w, h = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i]
            text_regions.append((x, y, w, h))
    return text_regions

# Function to check if a point (x, y) lies inside any contour
def is_point_inside_contour(x, y, contour):
    return cv2.pointPolygonTest(contour, (x, y), False) >= 0

# Function to determine if a region is covered by water based on the intersection of text regions and waterbody contours
def determine_water_coverage(image, text_regions, waterbody_contours):
    water_text_regions = []
    non_water_text_regions = []
    for region in text_regions:
        # Check if the center of the region lies inside any waterbody contour
        x, y, w, h = region
        region_center_x = x + w // 2
        region_center_y = y + h // 2
        water_coverage = False
        for contour in waterbody_contours:
            if is_point_inside_contour(region_center_x, region_center_y, contour):
                water_coverage = True
                break
        if water_coverage:
            water_text_regions.append(region)
        else:
            non_water_text_regions.append(region)
    return water_text_regions, non_water_text_regions

# Function to display the image with water coverage information
def display_image_with_water_coverage(image, water_text_regions, non_water_text_regions):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for region in water_text_regions:
        x, y, w, h = region
        plt.Rectangle((x, y), w, h, fill=None, edgecolor='blue', linewidth=2)  # Use plt.Rectangle instead of plt.Rectangle
    for region in non_water_text_regions:
        x, y, w, h = region
        plt.Rectangle((x, y), w, h, fill=None, edgecolor='red', linewidth=2)  # Use plt.Rectangle instead of plt.Rectangle
    plt.title("Text Regions Covered by Water (Blue) and Not Covered by Water (Red)")
    plt.axis('off')
    plt.show()

# Function to display the table of text regions covered by water or not
def display_text_region_table(water_text_regions, non_water_text_regions):
    print("Text Regions Covered by Water:")
    print("------------------------------")
    print("Left   |   Top   |   Width   |   Height   |   Covered by Water")
    print("---------------------------------------------------------------")
    for region in water_text_regions:
        print(f"{region[0]:<7} | {region[1]:<7} | {region[2]:<9} | {region[3]:<10} | {'Yes':<18}")
    print("\nText Regions Not Covered by Water:")
    print("----------------------------------")
    print("Left   |   Top   |   Width   |   Height")
    print("----------------------------------------")
    for region in non_water_text_regions:
        print(f"{region[0]:<7} | {region[1]:<7} | {region[2]:<9} | {region[3]:<10} | {'No':<18}")

# Assuming 'images' and 'changes' are available from previous processing
# Assuming 'reference_image' and 'biggest_change_image' are available

# Perform OCR on the image with the biggest change
biggest_change_image_with_water_text = images[max_change_idx + 1].copy()  # Make a copy to preserve original
text_data = perform_ocr(biggest_change_image_with_water_text)

# Extract bounding boxes of text regions
text_regions = get_text_regions(text_data)

# Find contours in the image with the biggest change
waterbody_contours_biggest_change = find_waterbody_contours(biggest_change_image_with_water_text)

# Determine if the detected text regions intersect with waterbody contours
water_text_regions, non_water_text_regions = determine_water_coverage(biggest_change_image_with_water_text, text_regions, waterbody_contours_biggest_change)

# Display the image with water coverage information
display_image_with_water_coverage(biggest_change_image_with_water_text, water_text_regions, non_water_text_regions)

# Display the table of text regions covered by water or not
display_text_region_table(water_text_regions, non_water_text_regions)


# In[8]:


import pytesseract
import matplotlib.pyplot as plt
import cv2

# Function to perform OCR on an image and return detected text and bounding boxes
def perform_ocr(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Perform thresholding to preprocess the image
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Perform OCR using pytesseract
    text_data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
    return text_data

# Function to filter out non-empty text regions and return their bounding boxes
def get_text_regions(text_data):
    text_regions = []
    for i in range(len(text_data['text'])):
        if text_data['text'][i] != '':
            x, y, w, h = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i]
            text_regions.append((x, y, w, h))
    return text_regions

# Function to check if a point (x, y) lies inside any contour
def is_point_inside_contour(x, y, contour):
    return cv2.pointPolygonTest(contour, (x, y), False) >= 0

# Function to determine if a region is covered by water based on the intersection of text regions and waterbody contours
def determine_water_coverage(image, text_regions, waterbody_contours):
    water_text_regions = []
    non_water_text_regions = []
    for region in text_regions:
        # Check if the center of the region lies inside any waterbody contour
        x, y, w, h = region
        region_center_x = x + w // 2
        region_center_y = y + h // 2
        water_coverage = False
        for contour in waterbody_contours:
            if is_point_inside_contour(region_center_x, region_center_y, contour):
                water_coverage = True
                break
        if water_coverage:
            water_text_regions.append(region)
        else:
            non_water_text_regions.append(region)
    return water_text_regions, non_water_text_regions

# Function to display the image with water coverage information
def display_image_with_water_coverage(image, water_text_regions, non_water_text_regions):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for region in water_text_regions:
        x, y, w, h = region
        plt.Rectangle((x, y), w, h, fill=None, edgecolor='blue', linewidth=2)  # Use plt.Rectangle instead of plt.Rectangle
    for region in non_water_text_regions:
        x, y, w, h = region
        plt.Rectangle((x, y), w, h, fill=None, edgecolor='red', linewidth=2)  # Use plt.Rectangle instead of plt.Rectangle
    plt.title("Text Regions Covered by Water (Blue) and Not Covered by Water (Red)")
    plt.axis('off')
    plt.show()

# Function to display the table of text regions covered by water or not
def display_text_region_table(water_text_regions, non_water_text_regions):
    print("Text Regions Covered by Water:")
    print("------------------------------")
    print("Left   |   Top   |   Width   |   Height   |   Covered by Water")
    print("---------------------------------------------------------------")
    for region in water_text_regions:
        x, y, w, h = region
        print(f"{x:<7} | {y:<7} | {w:<9} | {h:<10} | {'Yes':<18}")
    print("\nText Regions Not Covered by Water:")
    print("----------------------------------")
    print("Left   |   Top   |   Width   |   Height")
    print("----------------------------------------")
    for region in non_water_text_regions:
        x, y, w, h = region
        print(f"{x:<7} | {y:<7} | {w:<9} | {h:<10} | {'No':<18}")

# Assuming 'images' and 'changes' are available from previous processing
# Assuming 'reference_image' and 'biggest_change_image' are available

# Perform OCR on the image with the biggest change
biggest_change_image_with_water_text = images[max_change_idx + 1].copy()  # Make a copy to preserve original
text_data = perform_ocr(biggest_change_image_with_water_text)

# Extract bounding boxes of text regions
text_regions = get_text_regions(text_data)

# Find contours in the image with the biggest change
waterbody_contours_biggest_change = find_waterbody_contours(biggest_change_image_with_water_text)

# Determine if the detected text regions intersect with waterbody contours
water_text_regions, non_water_text_regions = determine_water_coverage(biggest_change_image_with_water_text, text_regions, waterbody_contours_biggest_change)

# Display the image with water coverage information
display_image_with_water_coverage(biggest_change_image_with_water_text, water_text_regions, non_water_text_regions)

# Display the table of text regions covered by water or not
display_text_region_table(water_text_regions, non_water_text_regions)


# In[ ]:





# In[9]:


import pytesseract
import matplotlib.pyplot as plt
import cv2

# Function to perform OCR on an image and return detected text and bounding boxes
def perform_ocr(image):
    # Invert the image
    inverted_image = cv2.bitwise_not(image)
    # Convert inverted image to grayscale
    gray_image = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2GRAY)
    # Perform thresholding to preprocess the image
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Perform OCR using pytesseract
    text_data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
    return text_data

# Function to filter out non-empty text regions and return their bounding boxes
def get_text_regions(text_data):
    text_regions = []
    for i in range(len(text_data['text'])):
        if text_data['text'][i] != '':
            x, y, w, h = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i]
            text_regions.append((x, y, w, h))
    return text_regions

# Function to check if a point (x, y) lies inside any contour
def is_point_inside_contour(x, y, contour):
    return cv2.pointPolygonTest(contour, (x, y), False) >= 0

# Function to determine if a region is covered by water based on the intersection of text regions and waterbody contours
def determine_water_coverage(image, text_regions, waterbody_contours):
    water_text_regions = []
    non_water_text_regions = []
    for region in text_regions:
        # Check if the center of the region lies inside any waterbody contour
        x, y, w, h = region
        region_center_x = x + w // 2
        region_center_y = y + h // 2
        water_coverage = False
        for contour in waterbody_contours:
            if is_point_inside_contour(region_center_x, region_center_y, contour):
                water_coverage = True
                break
        if water_coverage:
            water_text_regions.append(region)
        else:
            non_water_text_regions.append(region)
    return water_text_regions, non_water_text_regions

# Function to display the image with water coverage information
def display_image_with_water_coverage(image, water_text_regions, non_water_text_regions):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for region in water_text_regions:
        x, y, w, h = region
        plt.Rectangle((x, y), w, h, fill=None, edgecolor='blue', linewidth=2)  # Use plt.Rectangle instead of plt.Rectangle
    for region in non_water_text_regions:
        x, y, w, h = region
        plt.Rectangle((x, y), w, h, fill=None, edgecolor='red', linewidth=2)  # Use plt.Rectangle instead of plt.Rectangle
    plt.title("Text Regions Covered by Water (Blue) and Not Covered by Water (Red)")
    plt.axis('off')
    plt.show()

# Function to display the table of text regions covered by water or not
def display_text_region_table(water_text_regions, non_water_text_regions):
    if water_text_regions:
        print("Text Regions Covered by Water:")
        print("------------------------------")
        print("Left   |   Top   |   Width   |   Height   |   Covered by Water")
        print("---------------------------------------------------------------")
        for region in water_text_regions:
            x, y, w, h = region
            print(f"{x:<7} | {y:<7} | {w:<9} | {h:<10} | {'Yes':<18}")
    else:
        print("No text regions covered by water found.")
    if non_water_text_regions:
        print("\nText Regions Not Covered by Water:")
        print("----------------------------------")
        print("Left   |   Top   |   Width   |   Height")
        print("----------------------------------------")
        for region in non_water_text_regions:
            x, y, w, h = region
            print(f"{x:<7} | {y:<7} | {w:<9} | {h:<10} | {'No':<18}")
    else:
        print("No text regions not covered by water found.")

# Prompt the user to input the image path
image_path = input("Enter the path to the image: ")

# Read the image
biggest_change_image_with_water_text = cv2.imread(image_path)

# Perform OCR on the image
text_data = perform_ocr(biggest_change_image_with_water_text)

# Extract bounding boxes of text regions
text_regions = get_text_regions(text_data)

# Find contours in the image
# Assuming 'find_waterbody_contours' function is defined elsewhere
waterbody_contours_biggest_change = find_waterbody_contours(biggest_change_image_with_water_text)

# Determine if the detected text regions intersect with waterbody contours
water_text_regions, non_water_text_regions = determine_water_coverage(biggest_change_image_with_water_text, text_regions, waterbody_contours_biggest_change)

# Display the image with water coverage information
display_image_with_water_coverage(biggest_change_image_with_water_text, water_text_regions, non_water_text_regions)

# Display the table of text regions covered by water or not
display_text_region_table(water_text_regions, non_water_text_regions)

# Invert the colors of the image
inverted_image = cv2.bitwise_not(biggest_change_image_with_water_text)

# Perform OCR on the inverted image
text_data = perform_ocr(inverted_image)


# In[10]:


import pytesseract
import matplotlib.pyplot as plt
import cv2

# Function to perform OCR on an image and return detected text and bounding boxes
def perform_ocr(image):
    # Invert the image
    inverted_image = cv2.bitwise_not(image)
    # Convert inverted image to grayscale
    gray_image = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2GRAY)
    # Perform thresholding to preprocess the image
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Perform OCR using pytesseract
    text_data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
    return text_data

# Function to filter out non-empty text regions and return their bounding boxes
def get_text_regions(text_data):
    text_regions = []
    for i in range(len(text_data['text'])):
        if text_data['text'][i] != '':
            x, y, w, h = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i]
            text_regions.append((x, y, w, h))
    return text_regions

# Function to check if a point (x, y) lies inside any contour
def is_point_inside_contour(x, y, contour):
    return cv2.pointPolygonTest(contour, (x, y), False) >= 0

# Function to determine if a region is covered by water based on the intersection of text regions and waterbody contours
def determine_water_coverage(image, text_regions, waterbody_contours):
    water_text_regions = []
    non_water_text_regions = []
    for region in text_regions:
        # Check if the center of the region lies inside any waterbody contour
        x, y, w, h = region
        region_center_x = x + w // 2
        region_center_y = y + h // 2
        water_coverage = False
        for contour in waterbody_contours:
            if is_point_inside_contour(region_center_x, region_center_y, contour):
                water_coverage = True
                break
        if water_coverage:
            water_text_regions.append(region)
        else:
            non_water_text_regions.append(region)
    return water_text_regions, non_water_text_regions

# Function to display the image with water coverage information
def display_image_with_water_coverage(image, water_text_regions, non_water_text_regions):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for region in water_text_regions:
        x, y, w, h = region
        plt.Rectangle((x, y), w, h, fill=None, edgecolor='blue', linewidth=2)  # Use plt.Rectangle instead of plt.Rectangle
    for region in non_water_text_regions:
        x, y, w, h = region
        plt.Rectangle((x, y), w, h, fill=None, edgecolor='red', linewidth=2)  # Use plt.Rectangle instead of plt.Rectangle
    plt.title("Text Regions Covered by Water (Blue) and Not Covered by Water (Red)")
    plt.axis('off')
    plt.show()

# Function to display the table of text regions covered by water or not
def display_text_region_table(water_text_regions, non_water_text_regions):
    if water_text_regions:
        print("Text Regions Covered by Water:")
        print("------------------------------")
        print("Left   |   Top   |   Width   |   Height   |   Covered by Water")
        print("---------------------------------------------------------------")
        for region in water_text_regions:
            x, y, w, h = region
            print(f"{x:<7} | {y:<7} | {w:<9} | {h:<10} | {'Yes':<18} | {'Water':<18}")  # Updated to include the covered by water information
    else:
        print("No text regions covered by water found.")
    if non_water_text_regions:
        print("\nText Regions Not Covered by Water:")
        print("----------------------------------")
        print("Left   |   Top   |   Width   |   Height | Covered by Water")
        print("----------------------------------------")
        for region in non_water_text_regions:
            x, y, w, h = region
            print(f"{x:<7} | {y:<7} | {w:<9} | {h:<10} | {'No':<18} | {'Not Water':<18}")  # Updated to include the covered by water information
    else:
        print("No text regions not covered by water found.")

# Prompt the user to input the image path
image_path = input("Enter the path to the image: ")

# Read the image
biggest_change_image_with_water_text = cv2.imread(image_path)

# Perform OCR on the image
text_data = perform_ocr(biggest_change_image_with_water_text)

# Extract bounding boxes of text regions
text_regions = get_text_regions(text_data)

# Find contours in the image
# Assuming 'find_waterbody_contours' function is defined elsewhere
waterbody_contours_biggest_change = find_waterbody_contours(biggest_change_image_with_water_text)

# Determine if the detected text regions intersect with waterbody contours
water_text_regions, non_water_text_regions = determine_water_coverage(biggest_change_image_with_water_text, text_regions, waterbody_contours_biggest_change)

# Display the image with water coverage information
display_image_with_water_coverage(biggest_change_image_with_water_text, water_text_regions, non_water_text_regions)

# Display the table of text regions covered by water or not
display_text_region_table(water_text_regions, non_water_text_regions)

# Invert the colors of the image
inverted_image = cv2.bitwise_not(biggest_change_image_with_water_text)

# Perform OCR on the inverted image
text_data = perform_ocr(inverted_image)


# In[11]:


#"Left", "Top", "Width", and "Height" are the column headers used to describe the bounding boxes of text regions. Each column represents a specific attribute of the bounding box:

 #   "Left": The x-coordinate of the left side of the bounding box.
 #  "Top": The y-coordinate of the top side of the bounding box.
 #   "Width": The width of the bounding box.
 #   "Height": The height of the bounding box.

import pytesseract
import matplotlib.pyplot as plt
import cv2

# Function to perform OCR on an image and return detected text and bounding boxes
def perform_ocr(image):
    # Invert the image
    inverted_image = cv2.bitwise_not(image)
    # Convert inverted image to grayscale
    gray_image = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2GRAY)
    # Perform thresholding to preprocess the image
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Perform OCR using pytesseract
    text_data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
    return text_data

# Function to filter out non-empty text regions and return their bounding boxes
def get_text_regions(text_data):
    text_regions = []
    for i in range(len(text_data['text'])):
        if text_data['text'][i] != '':
            x, y, w, h = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i]
            text_regions.append((x, y, w, h, text_data['text'][i]))  # Include text in the tuple
    return text_regions

# Function to check if a point (x, y) lies inside any contour
def is_point_inside_contour(x, y, contour):
    return cv2.pointPolygonTest(contour, (x, y), False) >= 0

# Function to determine if a region is covered by water based on the intersection of text regions and waterbody contours
def determine_water_coverage(image, text_regions, waterbody_contours):
    water_text_regions = []
    non_water_text_regions = []
    for region in text_regions:
        # Check if the center of the region lies inside any waterbody contour
        x, y, w, h, text = region  # Unpack the text from the region tuple
        region_center_x = x + w // 2
        region_center_y = y + h // 2
        water_coverage = False
        for contour in waterbody_contours:
            if is_point_inside_contour(region_center_x, region_center_y, contour):
                water_coverage = True
                break
        if water_coverage:
            water_text_regions.append((x, y, w, h, text))  # Include text in the tuple
        else:
            non_water_text_regions.append((x, y, w, h, text))  # Include text in the tuple
    return water_text_regions, non_water_text_regions

# Function to display the image with water coverage information
def display_image_with_water_coverage(image, water_text_regions, non_water_text_regions):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for region in water_text_regions:
        x, y, w, h, _ = region  # Extract only the position information
        plt.Rectangle((x, y), w, h, fill=None, edgecolor='blue', linewidth=2)  # Use plt.Rectangle instead of plt.Rectangle
        plt.text(x + w, y + h, 'Water', color='blue')  # Display "Water" next to the region
    for region in non_water_text_regions:
        x, y, w, h, _ = region  # Extract only the position information
        plt.Rectangle((x, y), w, h, fill=None, edgecolor='red', linewidth=2)  # Use plt.Rectangle instead of plt.Rectangle
        plt.text(x + w, y + h, 'Not Water', color='red')  # Display "Not Water" next to the region
    plt.title("Text Regions Covered by Water (Blue) and Not Covered by Water (Red)")
    plt.axis('off')
    plt.show()

# Function to display the table of text regions covered by water or not
def display_text_region_table(water_text_regions, non_water_text_regions):
    if water_text_regions:
        print("Text Regions Covered by Water:")
        print("------------------------------")
        print("Left   |   Top   |   Width   |   Height   |   Covered by Water  |   Text")
        print("----------------------------------------------------------------------")
        for region in water_text_regions:
            x, y, w, h, text = region
            print(f"{x:<7} | {y:<7} | {w:<9} | {h:<10} | {'Yes':<18} | {text:<20}")  # Updated to include the covered by water information
    else:
        print("No text regions covered by water found.")
    if non_water_text_regions:
        print("\nText Regions Not Covered by Water:")
        print("----------------------------------")
        print("Left   |   Top   |   Width   |   Height | Covered by Water  |   Text")
        print("----------------------------------------")
        for region in non_water_text_regions:
            x, y, w, h, text = region
            print(f"{x:<7} | {y:<7} | {w:<9} | {h:<10} | {'No':<18} | {text:<20}")  # Updated to include the covered by water information
    else:
        print("No text regions not covered by water found.")

# Prompt the user to input the image path
image_path = input("Enter the path to the image: ")

# Read the image
biggest_change_image_with_water_text = cv2.imread(image_path)

# Perform OCR on the image
text_data = perform_ocr(biggest_change_image_with_water_text)

# Extract bounding boxes of text regions
text_regions = get_text_regions(text_data)

# Find contours in the image
# Assuming 'find_waterbody_contours' function is defined elsewhere
waterbody_contours_biggest_change = find_waterbody_contours(biggest_change_image_with_water_text)

# Determine if the detected text regions intersect with waterbody contours
water_text_regions, non_water_text_regions = determine_water_coverage(biggest_change_image_with_water_text, text_regions, waterbody_contours_biggest_change)

# Display the image with water coverage information
display_image_with_water_coverage(biggest_change_image_with_water_text, water_text_regions, non_water_text_regions)

# Display the table of text regions covered by water or not
display_text_region_table(water_text_regions, non_water_text_regions)

# Invert the colors of the image
inverted_image = cv2.bitwise_not(biggest_change_image_with_water_text)

# Perform OCR on the inverted image
text_data = perform_ocr(inverted_image)

import pytesseract
import matplotlib.pyplot as plt
import cv2

# Function to perform OCR on an image and return detected text and bounding boxes
def perform_ocr(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # Perform adaptive thresholding
    threshold_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Perform OCR using pytesseract
    text_data = pytesseract.image_to_data(threshold_image, output_type=pytesseract.Output.DICT, lang='eng', config='--psm 6')
    return text_data

# Function to filter out non-empty text regions and return their bounding boxes
def get_text_regions(text_data):
    text_regions = []
    for i in range(len(text_data['text'])):
        if text_data['text'][i].strip() != '':
            x, y, w, h = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i]
            text_regions.append((x, y, w, h))
    return text_regions

# Rest of the code remains the same...

# Perform OCR on the image
text_data = perform_ocr(biggest_change_image_with_water_text)



# In[12]:


import pytesseract
import matplotlib.pyplot as plt
import cv2

# Function to perform OCR on an image and return detected text and bounding boxes
def perform_ocr(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # Perform adaptive thresholding
    threshold_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Perform OCR using pytesseract
    text_data = pytesseract.image_to_data(threshold_image, output_type=pytesseract.Output.DICT, lang='eng', config='--psm 6')
    return text_data

# Function to filter out non-empty text regions and return their bounding boxes
def get_text_regions(text_data):
    text_regions = []
    for i in range(len(text_data['text'])):
        if text_data['text'][i].strip() != '':
            x, y, w, h = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i]
            text = text_data['text'][i].strip()
            text_regions.append((x, y, w, h, text))
    return text_regions

# Function to check if a point (x, y) lies inside any contour
def is_point_inside_contour(x, y, contour):
    return cv2.pointPolygonTest(contour, (x, y), False) >= 0

# Function to determine if a region is covered by water based on the intersection of text regions and waterbody contours
def determine_water_coverage(image, text_regions, waterbody_contours):
    water_text_regions = []
    non_water_text_regions = []
    for region in text_regions:
        # Check if the center of the region lies inside any waterbody contour
        x, y, w, h, text = region
        region_center_x = x + w // 2
        region_center_y = y + h // 2
        water_coverage = False
        for contour in waterbody_contours:
            if is_point_inside_contour(region_center_x, region_center_y, contour):
                water_coverage = True
                break
        if water_coverage:
            water_text_regions.append((x, y, w, h, text))
        else:
            non_water_text_regions.append((x, y, w, h, text))
    return water_text_regions, non_water_text_regions

# Function to display the image with water coverage information
def display_image_with_water_coverage(image, water_text_regions, non_water_text_regions):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for region in water_text_regions:
        x, y, w, h, _ = region
        plt.Rectangle((x, y), w, h, fill=None, edgecolor='blue', linewidth=2)  # Use plt.Rectangle instead of plt.Rectangle
    for region in non_water_text_regions:
        x, y, w, h, _ = region
        plt.Rectangle((x, y), w, h, fill=None, edgecolor='red', linewidth=2)  # Use plt.Rectangle instead of plt.Rectangle
    plt.title("Text Regions Covered by Water (Blue) and Not Covered by Water (Red)")
    plt.axis('off')
    plt.show()

# Function to display the table of text regions covered by water or not
def display_text_region_table(water_text_regions, non_water_text_regions):
    print("Text Regions:")
    print("--------------------------------------------------------------------------------")
    print("Location")
    print("--------------------------------------------------------------------------------")
    for region in water_text_regions:
        _, _, _, _, text = region
        print(text + " (Water)")
    for region in non_water_text_regions:
        _, _, _, _, text = region
        print(text + " (Not Water)")

# Prompt the user to input the image path
image_path = input("Enter the path to the image: ")

# Read the image
biggest_change_image_with_water_text = cv2.imread(image_path)

# Perform OCR on the image
text_data = perform_ocr(biggest_change_image_with_water_text)

# Extract bounding boxes of text regions
text_regions = get_text_regions(text_data)

# Find contours in the image
# Assuming 'find_waterbody_contours' function is defined elsewhere
waterbody_contours_biggest_change = find_waterbody_contours(biggest_change_image_with_water_text)

# Determine if the detected text regions intersect with waterbody contours
water_text_regions, non_water_text_regions = determine_water_coverage(biggest_change_image_with_water_text, text_regions, waterbody_contours_biggest_change)

# Display the image with water coverage information
display_image_with_water_coverage(biggest_change_image_with_water_text, water_text_regions, non_water_text_regions)

# Display the table of text regions covered by water or not
display_text_region_table(water_text_regions, non_water_text_regions)


# In[13]:


import pytesseract
import matplotlib.pyplot as plt
import cv2

# Function to perform OCR on an image and return detected text and bounding boxes
def perform_ocr(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # Perform adaptive thresholding
    threshold_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Perform OCR using pytesseract
    text_data = pytesseract.image_to_data(threshold_image, output_type=pytesseract.Output.DICT, lang='eng', config='--psm 6')
    return text_data

# Function to filter out non-empty text regions and return their bounding boxes
def get_text_regions(text_data):
    text_regions = []
    for i in range(len(text_data['text'])):
        if text_data['text'][i].strip() != '':
            x, y, w, h = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i]
            text = text_data['text'][i].strip()
            text_regions.append((x, y, w, h, text))
    return text_regions

# Function to check if a point (x, y) lies inside any contour
def is_point_inside_contour(x, y, contour):
    return cv2.pointPolygonTest(contour, (x, y), False) >= 0

# Function to determine if a region is covered by water based on the intersection of text regions and waterbody contours
def determine_water_coverage(image, text_regions, waterbody_contours):
    water_text_regions = []
    non_water_text_regions = []
    for region in text_regions:
        # Check if the center of the region lies inside any waterbody contour
        x, y, w, h, text = region
        region_center_x = x + w // 2
        region_center_y = y + h // 2
        water_coverage = False
        for contour in waterbody_contours:
            if is_point_inside_contour(region_center_x, region_center_y, contour):
                water_coverage = True
                break
        if water_coverage:
            water_text_regions.append((x, y, w, h, text))
        else:
            non_water_text_regions.append((x, y, w, h, text))
    return water_text_regions, non_water_text_regions

# Function to display the image with water coverage information
def display_image_with_water_coverage(image, water_text_regions, non_water_text_regions):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for region in water_text_regions:
        x, y, w, h, _ = region
        plt.Rectangle((x, y), w, h, fill=None, edgecolor='blue', linewidth=2)  # Use plt.Rectangle instead of plt.Rectangle
    for region in non_water_text_regions:
        x, y, w, h, _ = region
        plt.Rectangle((x, y), w, h, fill=None, edgecolor='red', linewidth=2)  # Use plt.Rectangle instead of plt.Rectangle
    plt.title("Text Regions Covered by Water (Blue) and Not Covered by Water (Red)")
    plt.axis('off')
    plt.show()

# Function to display the table of text regions covered by water or not
def display_text_region_table(water_text_regions, non_water_text_regions):
    print("Text Regions Covered by Water:")
    print("--------------------------------------")
    print("Location")
    print("--------------------------------------")
    for region in water_text_regions:
        _, _, _, _, text = region
        print(text)
    
    print("\nText Regions Not Covered by Water:")
    print("--------------------------------------")
    print("Location")
    print("--------------------------------------")
    for region in non_water_text_regions:
        _, _, _, _, text = region
        print(text)

# Prompt the user to input the image path
image_path = input("Enter the path to the image: ")

# Read the image
biggest_change_image_with_water_text = cv2.imread(image_path)

# Perform OCR on the image
text_data = perform_ocr(biggest_change_image_with_water_text)

# Extract bounding boxes of text regions
text_regions = get_text_regions(text_data)

# Find contours in the image
# Assuming 'find_waterbody_contours' function is defined elsewhere
waterbody_contours_biggest_change = find_waterbody_contours(biggest_change_image_with_water_text)

# Determine if the detected text regions intersect with waterbody contours
water_text_regions, non_water_text_regions = determine_water_coverage(biggest_change_image_with_water_text, text_regions, waterbody_contours_biggest_change)

# Display the image with water coverage information
display_image_with_water_coverage(biggest_change_image_with_water_text, water_text_regions, non_water_text_regions)

# Display the table of text regions covered by water or not
display_text_region_table(water_text_regions, non_water_text_regions)


# ##tests

# In[14]:


import pytesseract
import matplotlib.pyplot as plt
import cv2

# Function to perform OCR on an image and return detected text and bounding boxes
def perform_ocr(image):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # Perform adaptive thresholding
    threshold_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Perform OCR using pytesseract
    text_data = pytesseract.image_to_data(threshold_image, output_type=pytesseract.Output.DICT, lang='eng', config='--psm 6')
    return text_data

# Function to filter out non-empty text regions and return their bounding boxes
def get_text_regions(text_data):
    text_regions = []
    for i in range(len(text_data['text'])):
        if text_data['text'][i].strip() != '':
            x, y, w, h = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i]
            text = text_data['text'][i].strip()
            text_regions.append((x, y, w, h, text))
    return text_regions

# Function to check if a point (x, y) lies inside any contour
def is_point_inside_contour(x, y, contour):
    return cv2.pointPolygonTest(contour, (x, y), False) >= 0

# Function to determine if a region is covered by water based on the intersection of text regions and waterbody contours
def determine_water_coverage(image, text_regions, waterbody_contours):
    water_text_regions = []
    non_water_text_regions = []
    for region in text_regions:
        # Check if the center of the region lies inside any waterbody contour
        x, y, w, h, text = region
        region_center_x = x + w // 2
        region_center_y = y + h // 2
        water_coverage = False
        for contour in waterbody_contours:
            if is_point_inside_contour(region_center_x, region_center_y, contour):
                water_coverage = True
                break
        if water_coverage:
            water_text_regions.append((x, y, w, h, text))
        else:
            non_water_text_regions.append((x, y, w, h, text))
    return water_text_regions, non_water_text_regions

# Function to display the image with water coverage information
def display_image_with_water_coverage(image, water_text_regions, non_water_text_regions):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for region in water_text_regions:
        x, y, w, h, _ = region
        plt.Rectangle((x, y), w, h, fill=None, edgecolor='blue', linewidth=2)  # Use plt.Rectangle instead of plt.Rectangle
    for region in non_water_text_regions:
        x, y, w, h, _ = region
        plt.Rectangle((x, y), w, h, fill=None, edgecolor='red', linewidth=2)  # Use plt.Rectangle instead of plt.Rectangle
    plt.title("Text Regions Covered by Water (Blue) and Not Covered by Water (Red)")
    plt.axis('off')
    plt.show()

# Function to display the table of text regions covered by water or not
def display_text_region_table(water_text_regions, non_water_text_regions):
    print("Text Regions Covered by Water:")
    print("--------------------------------------")
    print("Location")
    print("--------------------------------------")
    for region in water_text_regions:
        _, _, _, _, text = region
        print(text)
    
    print("\nText Regions Not Covered by Water:")
    print("--------------------------------------")
    print("Location")
    print("--------------------------------------")
    for region in non_water_text_regions:
        _, _, _, _, text = region
        print(text)

# Prompt the user to input the image path
image_path = input("Enter the path to the image: ")

# Read the image
biggest_change_image_with_water_text = cv2.imread(image_path)

# Perform OCR on the image
text_data = perform_ocr(biggest_change_image_with_water_text)

# Extract bounding boxes of text regions
text_regions = get_text_regions(text_data)

# Find contours in the image
# Assuming 'find_waterbody_contours' function is defined elsewhere
waterbody_contours_biggest_change = find_waterbody_contours(biggest_change_image_with_water_text)

# Determine if the detected text regions intersect with waterbody contours
water_text_regions, non_water_text_regions = determine_water_coverage(biggest_change_image_with_water_text, text_regions, waterbody_contours_biggest_change)

# Display the image with water coverage information
display_image_with_water_coverage(biggest_change_image_with_water_text, water_text_regions, non_water_text_regions)

# Display the table of text regions covered by water or not
display_text_region_table(water_text_regions, non_water_text_regions)


# In[15]:


import pytesseract
import matplotlib.pyplot as plt
import cv2

# Function to perform OCR on an image and return detected text and bounding boxes
def perform_ocr(image):
    # Invert the image
    inverted_image = cv2.bitwise_not(image)
    # Convert inverted image to grayscale
    gray_image = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2GRAY)
    # Perform thresholding to preprocess the image
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Perform OCR using pytesseract
    text_data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
    return text_data

# Function to filter out non-empty text regions and return their bounding boxes
def get_text_regions(text_data, image_width, image_height):
    text_regions = []
    for i in range(len(text_data['text'])):
        if text_data['text'][i].strip() != '':
            x, y, w, h = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i]
            # Ensure the text region is within the image boundaries
            if 0 <= x < image_width and 0 <= y < image_height:
                text_regions.append((x, y, w, h))
    return text_regions

# Function to display the image with text regions highlighted
def display_image_with_text(image, text_regions):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for region in text_regions:
        x, y, w, h = region
        plt.Rectangle((x, y), w, h, fill=None, edgecolor='blue', linewidth=2)
    plt.title("Text Regions Detected in the Image")
    plt.axis('off')
    plt.show()

# Function to display the table of text regions
def display_text_region_table(text_regions):
    if text_regions:
        print("Detected Text Regions:")
        print("------------------------------")
        print("Left   |   Top   |   Width   |   Height")
        print("----------------------------------------")
        for region in text_regions:
            x, y, w, h = region
            print(f"{x:<7} | {y:<7} | {w:<9} | {h:<10}")
    else:
        print("No text regions detected.")

# Prompt the user to input the image path
image_path = input("Enter the path to the image: ")

# Read the image
image = cv2.imread(image_path)
image_height, image_width = image.shape[:2]

# Perform OCR on the image
text_data = perform_ocr(image)

# Extract bounding boxes of text regions
text_regions = get_text_regions(text_data, image_width, image_height)

# Display the image with text regions highlighted
display_image_with_text(image, text_regions)

# Display the table of text regions
display_text_region_table(text_regions)


# In[16]:


import pytesseract
import cv2
import numpy as np

# Function to perform OCR on an image and return detected text and bounding boxes
def perform_ocr(image):
    # Invert the image
    inverted_image = cv2.bitwise_not(image)
    # Convert inverted image to grayscale
    gray_image = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2GRAY)
    # Perform thresholding to preprocess the image
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Perform OCR using pytesseract
    text_data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
    return text_data

# Function to filter out non-empty text regions and return their bounding boxes
def get_text_regions(text_data, image_width, image_height):
    text_regions = []
    for i in range(len(text_data['text'])):
        if text_data['text'][i].strip() != '':
            x, y, w, h = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i]
            # Ensure the text region is within the image boundaries
            if 0 <= x < image_width and 0 <= y < image_height:
                text_regions.append((x, y, w, h))
    return text_regions

# Function to compare text regions between two images
def compare_text_regions(image1_text_regions, image2_text_regions):
    common_text_regions = []
    unique_to_image1 = []
    unique_to_image2 = []

    # Find common text regions
    for region in image1_text_regions:
        if region in image2_text_regions:
            common_text_regions.append(region)
        else:
            unique_to_image1.append(region)

    # Find text regions unique to each image
    for region in image1_text_regions:
        if region not in common_text_regions:
            unique_to_image1.append(region)

    for region in image2_text_regions:
        if region not in common_text_regions:
            unique_to_image2.append(region)

    return common_text_regions, unique_to_image1, unique_to_image2

# Prompt the user to input the paths to the images
image1_path = input("Enter the path to the first image: ")
image2_path = input("Enter the path to the second image: ")

# Read the images
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Perform OCR on both images
image1_text_data = perform_ocr(image1)
image2_text_data = perform_ocr(image2)

# Extract bounding boxes of text regions
image1_text_regions = get_text_regions(image1_text_data, image1.shape[1], image1.shape[0])
image2_text_regions = get_text_regions(image2_text_data, image2.shape[1], image2.shape[0])

# Compare text regions between the two images
common_text_regions, unique_to_image1, unique_to_image2 = compare_text_regions(image1_text_regions, image2_text_regions)

# Display results
print("Common Text Regions:")
print(common_text_regions)
print("\nText Regions Unique to Image 1:")
print(unique_to_image1)
print("\nText Regions Unique to Image 2:")
print(unique_to_image2)


# In[17]:


import pytesseract
import cv2
import numpy as np

# Function to perform OCR on an image and return detected text and bounding boxes
def perform_ocr(image):
    # Invert the image
    inverted_image = cv2.bitwise_not(image)
    # Convert inverted image to grayscale
    gray_image = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2GRAY)
    # Perform thresholding to preprocess the image
    _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Perform OCR using pytesseract
    text_data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT)
    return text_data

# Function to filter out non-empty text regions and return their bounding boxes
def get_text_regions(text_data, image_width, image_height):
    text_regions = []
    for i in range(len(text_data['text'])):
        if text_data['text'][i].strip() != '':
            x, y, w, h = text_data['left'][i], text_data['top'][i], text_data['width'][i], text_data['height'][i]
            # Ensure the text region is within the image boundaries
            if 0 <= x < image_width and 0 <= y < image_height:
                text_regions.append((x, y, w, h))
    return text_regions

# Function to check if a point (x, y) lies inside any contour
def is_point_inside_contour(x, y, contour):
    return cv2.pointPolygonTest(contour, (x, y), False) >= 0

# Function to determine if a region is covered by water based on the intersection of text regions and waterbody contours
def determine_water_coverage(image, text_regions, waterbody_contours):
    water_text_regions = []
    non_water_text_regions = []
    for region in text_regions:
        # Check if the center of the region lies inside any waterbody contour
        x, y, w, h = region
        region_center_x = x + w // 2
        region_center_y = y + h // 2
        water_coverage = False
        for contour in waterbody_contours:
            if is_point_inside_contour(region_center_x, region_center_y, contour):
                water_coverage = True
                break
        if water_coverage:
            water_text_regions.append(region)
        else:
            non_water_text_regions.append(region)
    return water_text_regions, non_water_text_regions

# Function to load waterbody contours from a file (assuming contours are stored as a list of numpy arrays)
def load_waterbody_contours(file_path):
    # Load contours from file
    contours = np.load(file_path, allow_pickle=True)
    return contours

# Function to compare text regions between two images
def compare_text_regions(image1_text_regions, image2_text_regions, image1_water_text_regions, image2_water_text_regions):
    common_text_regions = []
    unique_to_image1 = []
    unique_to_image2 = []
    water_text_regions = []

    # Find common text regions
    for region in image1_text_regions:
        if region in image2_text_regions:
            common_text_regions.append(region)
        else:
            unique_to_image1.append(region)

    # Find text regions unique to each image
    for region in image1_text_regions:
        if region not in common_text_regions:
            unique_to_image1.append(region)

    for region in image2_text_regions:
        if region not in common_text_regions:
            unique_to_image2.append(region)

    # Determine water coverage for common text regions
    for region in common_text_regions:
        if region in image1_water_text_regions and region in image2_water_text_regions:
            water_text_regions.append(region)

    return common_text_regions, unique_to_image1, unique_to_image2, water_text_regions

# Prompt the user to input the paths to the images
image1_path = input("Enter the path to the first image: ")
image2_path = input("Enter the path to the second image: ")

# Read the images
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Perform OCR on both images
image1_text_data = perform_ocr(image1)
image2_text_data = perform_ocr(image2)

# Extract bounding boxes of text regions
image1_text_regions = get_text_regions(image1_text_data, image1.shape[1], image1.shape[0])
image2_text_regions = get_text_regions(image2_text_data, image2.shape[1], image2.shape[0])

# Print information about loading waterbody contours
print("Before proceeding, please ensure you have the following files:")
print(f"- Waterbody contours file for {image1_path}:")
print(f"  {image1_path}_waterbody_contours.npy")
print(f"- Waterbody contours file for {image2_path}:")
print(f"  {image2_path}_waterbody_contours.npy")
print("Please make sure the contours files exist in the specified locations.")



# In[18]:


# Prompt the user to input the paths to the waterbody contours files
waterbody_contours_file1 = input("Enter the path to the waterbody contours file for the first image: ")
waterbody_contours_file2 = input("Enter the path to the waterbody contours file for the second image: ")

# Load waterbody contours for both images
waterbody_contours1 = load_waterbody_contours(waterbody_contours_file1)
waterbody_contours2 = load_waterbody_contours(waterbody_contours_file2)

# Determine water coverage for text regions in both images
image1_water_text_regions, _ = determine_water_coverage(image1, image1_text_regions, waterbody_contours1)
image2_water_text_regions, _ = determine_water_coverage(image2, image2_text_regions, waterbody_contours2)

# Compare text regions between the two images
common_text_regions, unique_to_image1, unique_to_image2, water_text_regions = compare_text_regions(image1_text_regions, image2_text_regions, image1_water_text_regions, image2_water_text_regions)

# Display results
print("\nCommon Text Regions:")
print(common_text_regions)
print("\nText Regions Unique to Image 1:")
print(unique_to_image1)
print("\nText Regions Unique to Image 2:")
print(unique_to_image2)
print("\nText Regions Under Water in Both Images:")
print(water_text_regions)

