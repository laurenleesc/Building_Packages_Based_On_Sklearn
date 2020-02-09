def lauren(n):
    ''' Returns n copies of 'Lauren ' as a single string'''
    return 'Lauren ' * n

def staples(s):
    ''' Counts instances of Lauren in string s'''
    return s.count('Lauren')

if __name__ == '__main__':
    # this lets us test our module
    s = lauren(3)
    print(s)
    print(staples(s))