import io
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import math
from fuzzywuzzy import fuzz
from google.oauth2 import service_account
import pandas as pd
from isbntools.app import *

def get_book_details(book):
    get_isbn = isbn_from_words(book)
    try:
        info = registry.bibformatters['labels'](meta(str(get_isbn)))
    except:
        info = 'No Information Available'
    return info

def get_infos(current_lib, current_shelf, loc):
    df = pd.read_csv('library_data/Master_data.csv',encoding = "ISO-8859-1" )
    image = cv2.imread(f'library_data/Libraries/{current_lib}/{loc}')
    new, points = draw_spine_lines(image)
    books = (df[(df.Library == current_lib) & (df.shelf == current_shelf)].Book.to_list())
    books = list(dict.fromkeys(books))
    infos = {}
    for book in books:
        print(book)
        info = find_book_in_library(book, current_lib, current_shelf)
        infos[book] = info
    print('exited')
    cv2.imwrite('static/display_shelf.jpg', new)
    return infos

def get_loc(current_lib, current_shelf):
    df = pd.read_csv('library_data/Master_data.csv',encoding = "ISO-8859-1" )
    loc = df[(df.Library == current_lib) & (df.shelf == current_shelf)].shelfImgLoc.to_list()[0]
    return (loc)
def find_presence_of_book(book_title, library, shelf, img):
# book_title = 'The Elements of Statistical learning'
# library = 'Crossword'
# shelf = 'Self_help'
    isPresent = False
    # print(isPresent)
    found = find_requested2(shelf, book_title, library, img)
    print('-------------------Found------------------', found)
    if found == '':
        info, shelf = find_missing(shelf, library, book_title)

        print(f'Book found in {shelf}')
        if shelf != '':
            isPresent = True
            details = [library, shelf]
        else:
            details = [library, 'Missing from all Shelves']
        return isPresent, details
    else:
        isPresent = True
        details = [library, shelf]
    return isPresent, details

def find_requested2(shelf, book_title, library, img):
    
    info = ''
    print(info)
    df = pd.read_csv('library_data/Master_data.csv',encoding = "ISO-8859-1" )
    final_image = resize_img(img)
    final_points = detect_spines(final_image)
    image = resize_img(img)
    y_max, _, _ = image.shape
    last_x1 = 0
    last_x2 = 0
#     prev_x1 = 0
#     prev_y1 = 0
#     prev_x2 = 0
#     prev_y2 = 0
    for point in final_points:
        ((x1, y1), (x2, y2)) = point
        crop_points = np.array([[last_x1, y_max],
                                [last_x2, 0],
                                [x2, y2],
                                [x1, y1]])
        # Crop the bounding rect
        rect = cv2.boundingRect(crop_points)
        x, y, w, h = rect
        cropped = image[y: y + h, x: x + w].copy()
        # make mask
        crop_points = crop_points - crop_points.min(axis=0)
        mask = np.zeros(cropped.shape[:2], np.uint8)
        cv2.drawContours(mask, [crop_points], -1, (255, 255, 255), -1, cv2.LINE_AA)
        # do bit-op
        dst = cv2.bitwise_and(cropped, cropped, mask=mask)
#         cropped_images.append(dst)
        cv2.imwrite('static/read.jpg', dst)
        text = detect_text_helper()
        print(text)
        if text is None:
            continue
        score = compute_score(book_title.lower(), text.lower())
        print(f"Book Title: {book_title}")
        print(f"Read Title: {text}")
        print(f"Score: {score}")
        print('----------------------------')
        if score>75:
            print(f'Book {book_title} found!')
            info = get_book_details(book_title)
            
            return info
        last_x1 = x1
        last_x2 = x2
#         
    print('Book not Found!')
    print('Locating book in other shelves!')
                      
    return info



def find_requested(shelf, book_title, library):
    
    info = ''
    df = pd.read_csv('library_data/Master_data.csv',encoding = "ISO-8859-1" )
#     print(library, shelf)
    loc = df[(df.Library == library) & (df.shelf == shelf)].shelfImgLoc.to_list()[0]
    img = cv2.imread(f"library_data/Libraries/{library}/{loc}")
    final_image = resize_img(img)
    final_points = detect_spines(final_image)
    image = resize_img(img)
    y_max, _, _ = image.shape
    last_x1 = 0
    last_x2 = 0
#     prev_x1 = 0
#     prev_y1 = 0
#     prev_x2 = 0
#     prev_y2 = 0
    for point in final_points:
        print(point)
        ((x1, y1), (x2, y2)) = point
        crop_points = np.array([[last_x1, y_max],
                                [last_x2, 0],
                                [x2, y2],
                                [x1, y1]])
        # Crop the bounding rect
        rect = cv2.boundingRect(crop_points)
        print(rect)
        x, y, w, h = rect
        cropped = image[y: y + h, x: x + w].copy()
        # make mask
        crop_points = crop_points - crop_points.min(axis=0)
        mask = np.zeros(cropped.shape[:2], np.uint8)
        cv2.drawContours(mask, [crop_points], -1, (255, 255, 255), -1, cv2.LINE_AA)
        # do bit-op
        dst = cv2.bitwise_and(cropped, cropped, mask=mask)
#         cropped_images.append(dst)
        print(dst)
        cv2.imwrite('static/read.jpg', dst)
        text = detect_text_helper()
        if text is None:
            continue
        score = compute_score(book_title.lower(), text.lower())
        print(f"Book Title: {book_title}")
        print(f"Read Title: {text}")
        print(f"Score: {score}")
        print('----------------------------')
        if score>75:
            print(f'Book {book_title} found!')
            info = get_book_details(book_title)
            
            return info
        last_x1 = x1
        last_x2 = x2
#         
    print('Book not Found!')
    print('Locating book in other shelves!')
                      
    return info


def find_missing(ogshelf, library, book_title):
    
    df = pd.read_csv('library_data/Master_data.csv',encoding = "ISO-8859-1" )
    shelves = list(set(df[(df.Library == library)].shelf.to_list()))
#     ogshelf = df[(df.Book == book_title)].shelfImgLoc.to_list()[0]
    
    for shelf in shelves:
        if shelf == ogshelf:
            continue
        img_loc = df[(df.Library == library) & (df.shelf == shelf)].shelfImgLoc.to_list()[0]
        img = cv2.imread(f'library_data/Libraries/{library}/{img_loc}')
        info = find_requested(shelf, book_title, library)
        if info is not '':
            return info, shelf
        
    return "Book Not Found in any shelf", ''


def find_book_in_library(book_title, library, shelf):


# book_title = 'The Elements of Statistical learning'
# library = 'Crossword'
# shelf = 'Self_help'
    found = find_requested(shelf, book_title, library)
    if found == '':
        info, new_shelf = find_missing(shelf, library, book_title)

        print(f'Book found in {new_shelf}')
        df = pd.read_csv('library_data/Master_data.csv',encoding = "ISO-8859-1" )
        shelfloc = list(set(df[(df.Library == library) & (df.shelf == new_shelf)].shelfImgLoc.to_list()))[0]
        img = cv2.imread(f'library_data/Libraries/{library}/{shelfloc}')
        draw_requested(img, book_title)
        return info

    return found
"""Detects text in the file."""
def detect_text_helper():
    from google.cloud import vision
    path = 'static/read.jpg'
    images=cv2.imread(path) 
    out=cv2.transpose(images)
    out=cv2.flip(out,flipCode=0)
    gray=cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) 
    cv2.imwrite('static/flipped.jpg', gray)
    creds = service_account.Credentials.from_service_account_file('/Users/seema/Downloads/OCRshelf-cd8e3a983d34.json')
    client = vision.ImageAnnotatorClient(credentials=creds,)


    with io.open('static/flipped.jpg', 'rb') as image_file:
         content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    if len(texts) > 0:
        text = response.text_annotations[0].description
    else:
        text = None
    return text

def draw_requested(img, book_title):
    
    final_image = resize_img(img)
    final_points = detect_spines(final_image)
    image = resize_img(img)
    y_max, _, _ = image.shape
    last_x1 = 0
    last_x2 = 0
#     prev_x1 = 0
#     prev_y1 = 0
#     prev_x2 = 0
#     prev_y2 = 0
    for point in final_points:
        ((x1, y1), (x2, y2)) = point
        crop_points = np.array([[last_x1, y_max],
                                [last_x2, 0],
                                [x2, y2],
                                [x1, y1]])
        print(crop_points)
        # Crop the bounding rect
        rect = cv2.boundingRect(crop_points)
        x, y, w, h = rect
        cropped = image[y: y + h, x: x + w].copy()
        # make mask
        crop_points = crop_points - crop_points.min(axis=0)
        mask = np.zeros(cropped.shape[:2], np.uint8)
        cv2.drawContours(mask, [crop_points], -1, (255, 255, 255), -1, cv2.LINE_AA)
        # do bit-op
        dst = cv2.bitwise_and(cropped, cropped, mask=mask)
#         cropped_images.append(dst)
        
        cv2.imwrite('static/read.jpg', dst)
        text = detect_text_helper()
        if text is None:
            continue
        score = compute_score(book_title.lower(), text.lower())
        print(f"Book Title: {book_title}")
        print(f"Read Title: {text}")
        print(f"Score: {score}")
        print('----------------------------')
        if score>75:
            final_image = cv2.line(final_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            final_image = cv2.line(final_image, (last_x2, 0), (last_x1, y_max), (0, 0, 255), 3)
            return final_image
        last_x1 = x1
        last_x2 = x2
#         
        
    return final_image


def compute_score(book_title, read_title):
    #, threshold = 75
    Ratio = fuzz.partial_ratio(book_title.lower(),read_title.lower()) 
    return (Ratio)
    


def remove_duplicate_lines(sorted_points):
    '''
    Serches for the lines that are drawn
    over each other in the image and returns
    a list of non duplicate line co-ordinates
    '''
    last_x1 = 0
    non_duplicate_points = []
    for point in sorted_points:
        ((x1, y1), (x2, y2)) = point
        if last_x1 == 0:
            non_duplicate_points.append(point)
            last_x1 = x1

        elif abs(last_x1 - x1) >= 25:
            non_duplicate_points.append(point)
            last_x1 = x1

    return non_duplicate_points


def get_points_in_x_and_y(hough_lines, max_y):
    '''
    Takes a list of trigonometric form of lines
    and returns their starting and ending
    co-ordinates
    '''
    points = []
    for line in hough_lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + (max_y + 100) * (-b))
        y1 = int(y0 + (max_y + 100) * (a))
        start = (x1, y1)

        x2 = int(x0 - (max_y + 100) * (-b))
        y2 = int(y0 - (max_y + 100) * (a))
        end = (x2, y2)

        points.append((start, end))

    # Add a line at the very end of the image
    points.append(((500, max_y), (500, 0)))

    return points


def shorten_line(points, y_max):
    '''
    Takes a list of starting and ending
    co-ordinates of different lines
    and returns their trimmed form matching
    the image height
    '''
    shortened_points = []
    for point in points:
        ((x1, y1), (x2, y2)) = point

        # Slope
        try:
            m = (y2 - y1) / (x2 - x1)
        except ZeroDivisionError:
            m = -1  # Infinite slope

        if m == -1:
            shortened_points.append(((x1, y_max), (x1, 0)))
            continue

        # From equation of line:
        # y-y1 = m (x-x1)
        # x = (y-y1)/m + x1
        # let y = y_max
        new_x1 = math.ceil(((y_max - y1) / m) + x1)
        start_point = (abs(new_x1), y_max)

        # Now let y = 0
        new_x2 = math.ceil(((0 - y1) / m) + x1)
        end_point = (abs(new_x2), 0)

        shortened_points.append((start_point, end_point))

    return shortened_points


def get_cropped_images(image, points):
    '''
    Takes a spine line drawn image and
    returns a list of opencv images splitted
    from the drawn lines
    '''
    image_og = image.copy()
    image = resize_img(image)
    y_max, _, _ = image.shape
    last_x1 = 0
    last_x2 = 0
    cropped_images = []

    for point in points:
        ((x1, y1), (x2, y2)) = point

        crop_points = np.array([[last_x1, y_max],
                                [last_x2, 0],
                                [x2, y2],
                                [x1, y1]])
        print(crop_points)
        # Crop the bounding rect
        rect = cv2.boundingRect(crop_points)
        x, y, w, h = rect
        cropped = image[y: y + h, x: x + w].copy()
        # make mask
        crop_points = crop_points - crop_points.min(axis=0)
        mask = np.zeros(cropped.shape[:2], np.uint8)
        cv2.drawContours(mask, [crop_points], -1, (255, 255, 255), -1, cv2.LINE_AA)
        # do bit-op
        dst = cv2.bitwise_and(cropped, cropped, mask=mask)
        cropped_images.append(dst)
        last_x1 = x1
        last_x2 = x2
        
    return cropped_images


def resize_img(img):
    img = img.copy()
    img_ht, img_wd, _ = img.shape
    ratio = img_wd / img_ht
    new_width = 500
    new_height = math.ceil(new_width / ratio)
    resized_image = cv2.resize(img, (new_width, new_height))

    return resized_image


def detect_spines(img):
    '''
    Returns a list of lines seperating
    the detected spines in the image
    '''
    img = img.copy()
    height, width, _ = img.shape

    blur = cv2.GaussianBlur(img, (5, 5), 0)

    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    edge = cv2.Canny(gray, 50, 70)

    # kernel = np.ones((4, 1), np.uint8)
    kernel = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype=np.uint8)

    img_erosion = cv2.erode(edge, kernel, iterations=1)

    lines = cv2.HoughLines(img_erosion, 1, np.pi / 180, 100)
    if lines is None:
        return []
    points = get_points_in_x_and_y(lines, height)
    points.sort(key=lambda val: val[0][0])
    non_duplicate_points = remove_duplicate_lines(points)

    final_points = shorten_line(non_duplicate_points, height)

    return final_points


def get_spines(img):
    
    final_image = resize_img(img)
    final_points = detect_spines(final_image)
    cropped_images = get_cropped_images(final_image, final_points)

    django_cropped_images = []
    for cropped_image in cropped_images:
        django_cropped_images.append(cropped_image)

    return django_cropped_images


def draw_spine_lines(img):
    
    final_image = resize_img(img)
    final_points = detect_spines(final_image)

    for point in final_points:
        ((x1, y1), (x2, y2)) = point
        final_image = cv2.line(final_image, (x1, y1), (x2, y2), (0, 0, 255), 3)

    return final_image, final_points