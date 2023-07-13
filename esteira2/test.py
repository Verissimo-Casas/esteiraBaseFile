import cv2 
import numpy as np 
   
def emptyFunction(): 
    pass
   
def main(): 
      
    
    image = np.zeros((512, 512, 3), np.uint8)  
    windowName ="Open CV Color Palette"
      
    
    cv2.namedWindow(windowName)  
       
    
    
    cv2.createTrackbar('Blue', windowName, 0, 255, emptyFunction) 
    cv2.createTrackbar('Green', windowName, 0, 255, emptyFunction) 
    cv2.createTrackbar('Red', windowName, 0, 255, emptyFunction) 
       
    
    
    while(True): 
        cv2.imshow(windowName, image) 
        if cv2.waitKey(1) == 27: 
            break
        
        blue = cv2.getTrackbarPos('Blue', windowName) 
        green = cv2.getTrackbarPos('Green', windowName) 
        red = cv2.getTrackbarPos('Red', windowName) 
        
        image[:] = [blue, green, red] 
        print(blue, green, red) 
           
    cv2.destroyAllWindows() 
if __name__=="__main__": 
    main() 