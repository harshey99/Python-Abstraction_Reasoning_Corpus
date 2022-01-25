#!/usr/bin/python

# ASSIGNMENT -3 (1MAI)
# GROUP SUBMISSION
# STUDENT - HUZEFA MANSOOR LOKHANDWALA (21241263)
# STUDENT - HARSHITHA BENGALURU RAGHURAM (21235396)
# GITHUB REPO - https://github.com/huzz/ARC

'''
 **Python features and libraries used and their commonalities and differences**

 Library - NUMPY
 Numpy is an in built opensource package in python which is used for scientific computing. 
 Here numpy is mainly used for Array manupilation and generation.
 The numpy functions below are explained in detail

1. np.where()
This function is used to return cordinates depending on the condition.
It can also modify values of the array depending on the condition
This function was used in all 4 tasks in two ways:
 a. to find the coordinates
 b. to modify the value of the array on basis of a condition
The common usage was to find the location of a particular colour value in an array and 
to replace a value to set a background colour(Task 1) or a specific pattern colour (Task 2)

2. np.pad()
This function is used to add padding. We can specify the pad value and pad width and on which axis pad is required.
This function was used in Task 1 and Task 2.
For Task 1 this was used to add pad to convert the colour array to the ouput dimension
For Task 2 this was used to make the grid pattern symmetrical

3. np.argmax() and np.argmin() 
Used to find the min and max value of an array
argmax was used in Task 1 and argmin was used in Task 3
Both were used to find the max and min value respectively.

4. arr.shape 
Used to find the shape of the array in (row, column)
This was used in all tasks.

5. np.flip 
This function is used to reverse the elements and get the mirror image. 
This function was used in Task 3 only.

6. np.triu 
This function is used to get an upper triangle of the array. The array is split on the diagonal and all 
values above the diagonal are retained and all values below are assigned as 0.
This function was used in Task 2 only.

7. np.rot90 
This function is to rotate the array by 90. We can specify how many times we want to rotate is by.
This fucntion was used in Task 2. If we use rot90 2 times this is the same as using the np.flip function

8. arr.copy() 
This is used to deepcopy the input array. 
This function is used in Task 4.

9 array slicing 
Normal Array slicing was used in multiple places. It has a format of: 
Arr[row_index1:row_index2, column_index1:column_index2]
This function is used in all tasks to generate or slice a new array.

10. np.unique() 
This function is used to return a list of all unique values present in the array. If return_counts=True, 
then it also returns the count for each unique element.
This function was used in Task 1 and Task 3 for getting the unique elemnts and their counts.

11. np.zeros() 
Creates an array of given shape of values all zero. This was used mainly to create output array with given dimensions.
This function was used in Task 1 and Task 3 for the same usage as mentioned above.

References:
https://numpy.org/doc/stable/numpy-user.pdf
Usage Date - 26/11/2021

https://numpy.org/doc/stable/user/quickstart.html
Usage Date - 26/11/2021
'''

import os, sys
import json
import numpy as np
import re

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.


#TASK 1
def solve_c8cbb738(x):
    # LEVEL : DIFFICULT
    #file: solve_c8cbb738.json

    '''
    Logic: 
    First we found the background colour add list and counts of the unique colours. 
    Then we found out the output array dimensions and initialized it to 0 values. 
    This could be found using np.where by finsing the locations of all colours and returning the maximum grid size.
    Once we got this, we then iterate thorugh the colours and find the colour grid with the background respectively. 
    The background is then filtered such that any value except of the colour is turned 0. If the shape of the colour 
    array is not the same as the ouput then we add a padding of row/column depending on the arragement on either side.
    Now we can simply add the colour array with the ouput array. Hence we get the right ouput.

    All the training and test sets have been solved successfully using this algorithm.
    '''
    
    #Get the the unique colour numbers and the counts of each colour
    (colour_list, colour_counts)=np.unique(x, return_counts=True)
    #Find the colour that has maximum count, which is our background colour
    background_colour = colour_list[np.argmax(colour_counts)]
    #Get the dimension of our output array. 
    #Iterating through the unique colours except the background colour and returning the dimension wof maximum colour
    dimension = max([(max(np.where(x==colour)[0])-min(np.where(x==colour)[0])) for colour in colour_list if colour!=background_colour]) + 1
    #cretaing an np array of zeros with output dimension
    output_array = np.zeros((dimension,dimension))
    #for each unique colour
    for colour in colour_list:
        #except the background colour
        if colour!=background_colour:
            #getting the colour row and column coordinates
            colour_row,colour_column  = np.where(x==colour)
            #slicing the array with the min and max coordinates of the colour
            #here we get the sliced array according to the colour coordinates
            colour_array = x[min(colour_row):(max(colour_row) + 1),min(colour_column):(max(colour_column) + 1)]
            #making every pixel in the colour_array '0' if its not of the main colour
            colour_array=np.where(colour_array!=colour, 0, colour_array)      
            #if the rows of the colour are less than rows of the output array
            if colour_array.shape[0]<output_array.shape[0]:
                #finding the pad width inorder to add inorder to make the dimension same as the ouput array
                pad_width = (output_array.shape[0]-colour_array.shape[0])-(output_array.shape[0]-colour_array.shape[0])//2
                #adding a pad on the top and bottom of the array
                colour_array = np.pad(colour_array, ((pad_width, pad_width), (0, 0)), mode='constant')
            #if the columns of the colour are less than columns of the output array
            elif colour_array.shape[1]<output_array.shape[1]:
                #finding the pad width inorder to add inorder to make the dimension same as the ouput array
                pad_width = (output_array.shape[1]-colour_array.shape[1])-(output_array.shape[1]-colour_array.shape[1])//2
                #adding a pad on the left and right of the array
                colour_array = np.pad(colour_array, ((0, 0), (pad_width, pad_width)), mode='constant')
            #simply adding the colour array to the output array 
            output_array+=colour_array
                
    #replacing all "0" with the background colour
    output_array=np.where(output_array==0, background_colour, output_array)
    
    #returning the final output value
    return output_array

#TASK 2
def solve_1b60fb0c(x):
    # LEVEL : DIFFICULT
    #file: solve_1b60fb0c.json

    '''
    Logic: 
    Here we try to get the reflection of the array across the diagonal, 
    But since the inputs are not symmetrical, we add padding to make it symmetrical
    First we find the first row, last row and last column values. If the first row is not equal to the 
    last row then we can say it is not symmetrical, also if the last column is al 0's it is observed 
    that the grid is then symetrical. Using this analysis we add padding to the upper_right/lower_right/upper/lower.
    Flags are set to track where the padding was applied. 
    Then we create an upper triangle grid from the input and rotate it twice by 90 degrees.
    This gives us the array except of the shape marked in red. This is obtained by taking the difference from the original 
    input. AT the end we just remove the extra padding added depending on the flag set.

    All the training and test sets have been solved successfully using this algorithm.
    '''
    
    #storing the original dimensions of x
    ori_dim=x.shape
    #storing the first row, last row and last column
    first_row = x[0]
    last_row = x[-1]
    last_column = x[:,-1]
    
    #initializing the pad_type as None
    #pad type can have 4 values ["upper_right", "lower_right", "upper", "lower"]
    pad_type=None
    # if the top row and bottom row are not equal and the last column is not full of 0 values
    if len(set(first_row))!=len(set(last_row)) and list(set(last_column))!=[0]:
        # if the first row is not full of 0 values and setting the respective flagz
        if list(set(first_row))!=[0]:
            #adding a pad to the top row and last column and setting the respective flag
            x=np.pad(x,((1,0),(0,1)))
            pad_type="upper_right"
        # if the last row is not full of 0 values
        else:
            #adding a pad to the bottom row and last column and setting the respective flag
            x=np.pad(x,((0,1),(0,1)))
            pad_type="lower_right"
    # if the top row and bottom row are not equal and the last column is full of 0 values
    elif len(set(first_row))!=len(set(last_row)) and list(set(last_column))==[0]:
        # if the first row is not full of 0 values
        if list(set(first_row))!=[0]:
            #adding a pad to the top row and setting the respective flag
            x=np.pad(x,((1,0),(0,0)))
            pad_type="upper"
        # if the last row is not full of 0 values
        else:
            #adding a pad to the bottom row and setting the respective flag
            x=np.pad(x,((0,1),(0,0)))
            pad_type="lower"
        
    #getting Upper triangle of an array and the lower part values are set to "0"
    diag_up = np.triu(x)
    #rotating the array 2 times by 0 degrees
    diag_up_T = np.rot90(diag_up, 2)
    #adding the diagonal upper triangle array and the rotated version
    output_array = diag_up+diag_up_T
    #replacing the values having value 2 with 1. 
    #This will make the entire shape except the red values
    output_array=np.where(output_array==2, 1, output_array)         
    #removing the difference with the original array to get the positions of red values
    diff_array=output_array-x
    #adding the difference array with the output array
    output_array+=diff_array
    
    #Removing the additional pads depending on the flag set
    if pad_type=='upper_right':output_array=output_array[output_array.shape[0]-ori_dim[0]:,:-(output_array.shape[1]-ori_dim[1])]
    elif pad_type=='lower_right':output_array=output_array[:-(output_array.shape[0]-ori_dim[0]),:-(output_array.shape[1]-ori_dim[1])]
    elif pad_type=='upper':output_array=output_array[(output_array.shape[0]-ori_dim[0]):,:]
    elif pad_type=='lower':output_array=output_array[:-(output_array.shape[0]-ori_dim[0]),:]
        
    return output_array

#TASK 3         
def solve_ff805c23(x):
    # LEVEL : MEDIUM TO DIFFICULT
    #file: ff805c23.json

    '''
    Logic:
    First we find the unique colours and thier counts. This colour with least count will be the box 
    whose pattern has to be replaced and returned. An output array of 0 values is created using the obtained box dimensions.
    The input array is flipped and the coordinates of the box colour is found. 
    Using these coordinates the input array is sliced to obatain the pattern.
    The pattern is then rotated and returned.

    All the training and test sets have been solved successfully using this algorithm.
    '''
    
    #Get the the unique colour numbers and the counts of each colour
    (colour_list, colour_counts)=np.unique(x, return_counts=True)
    #Find the colour that has minimum count, which is our colour for our box
    box_colour = colour_list[np.argmin(colour_counts)]
    #findind the dimension of the ouputarray
    dimension = (max(np.where(x==box_colour)[0])-min(np.where(x==box_colour)[0]))+1
    #cretaing an np array of zeros with output dimension
    output_array = np.zeros((dimension,dimension))
    #fliiping the the array to get a mirror image 
    flip_array = np.flip(x)
    #in the flipped array we are finding the box cordinates
    colour_row,colour_column  = np.where(flip_array==box_colour)
    #getting the pattern using the above cordinates by slicing the original array
    pattern = x[min(colour_row):max(colour_row)+1, min(colour_column):max(colour_column)+1] 
    
    #returning the flipped version of the pattern
    return np.flip(pattern)

#TASK 4
def solve_3631a71a(x):
    # LEVEL : MEDIUM TO DIFFICULT
    # file: 3631a71a.json

    '''
    Logic:
    FIrst we get all the row and column coordinates where the colour is of 9 value.
    On iterating through these coordinates, we replace the value of our copied array
    from the input. If the colour of value does not exists we simply assign the values 
    as it is for that row column slice. If the value 9 exists we then assign the mirror 
    vertical or mirror horizontal values of the input depending on the previous coordinate.

    All the training and test sets have been solved successfully using this algorithm.
    '''
    
    #creating a copy of the input array, to retain the original input array
    x_copy = x.copy() 
    # finding the row and columns where the colour in x is 9
    colour_row, colour_col = np.where(x == 9) 

    #Iterating through the coordinates where the colour is 9
    for row, column in zip(colour_row, colour_col):
        # If x[column, row] is not 9, replace x_copy[row, column] with x[column, row]
        if x[column, row] != 9:
            x_copy[row, column] = x[column, row]
        else:
            # Else if the mirror value (vertical) is not equal to 9, replace x_copy[row, column] with x[row, 1 - column]
            if x[row, 1 - column] != 9:
                x_copy[row, column] = x[row, 1 - column] 
            #Else replace x_copy[row, column] with the mirror value (horizontal), which is x[1 - row, column]
            else:
                x_copy[row, column] = x[1 - row, column]
    
    #returning the final array
    return x_copy

def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)
    
def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    if y.shape != yhat.shape:
        print(f"False. Incorrect shape: {y.shape} v {yhat.shape}")
    else:
        print(np.all(y == yhat))


if __name__ == "__main__": main()

