import torch
"""
z = torch.zeros(5,3)
#creates a tesnor of 5 rows and 3 columns, .zeros() fills with zeroes
#tensor in simple terms multi dimensional array/matrix

print(z)
print(z.dtype)
#default is 32 floating points
#prints the tensor value types

"""

"""
i = torch.ones((5,3), dtype=torch.int16)
#fills with ones, dtype determines the value type
print(i)

torch.manual_seed(1729)
r1 = torch.rand(2, 2)
#generates a random tensor of 2 x 2 dimension from seed 1729
print(r1)

r2 = torch.rand(2,2)
#re-seeds for a new seed to generate new random numbers
print (r2)

torch.manual_seed(1729)
#Will match r1
r3 = torch.rand(2,2)
print(r3)
"""

r = torch.rand(2,2) * -1
print(r)

#Absolute Value
print('\n')
print(torch.abs(r))