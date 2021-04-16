import cv2
import pandas as pd
from bookLines import *
from isbntools.app import *
def get_books():
    master_df = pd.read_csv('library_data/Master_data.csv',encoding = "ISO-8859-1" )
    book_list = list(set(master_df.Book.to_list()))
    return book_list

def get_book_details(book):
    get_isbn = isbn_from_words(book)
    try:
        info = registry.bibformatters['labels'](meta(str(get_isbn)))
    except:
        info = 'No Information Available'
    return info

def get_book_pic(book):
    master_df = pd.read_csv('library_data/Master_data.csv',encoding = "ISO-8859-1" )
    book_data = master_df[master_df.Book == book]
    Library = book_data.Library.to_list()[0]
    Shelf = book_data.shelf.to_list()[0]
    loc = book_data.shelfImgLoc.to_list()[0]
    shelf_path = f'library_data/Libraries/{Library}/{loc}'
    shelf_path = shelf_path.replace(' ', '_')
    print(shelf_path)
    img = cv2.imread(shelf_path)
    img = draw_requested(img, book)
    print(img.shape)
    filename = f'{Library}_{Shelf}_{book}.jpg'
    filename = filename.replace(' ', '_')
    cv2.imwrite(f'static/{filename}', img)
    book_deets = {
        'filename': filename,
        'library': Library,
        'shelf': Shelf,        
    }
    return book_deets

