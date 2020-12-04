def printDivisors(n) : 
	i = 1
	while i <= n : 
		if (n % i==0) : 
			print(i), 
		i = i + 1
		
# Driver method 
print("The divisors of 100 are: ")
printDivisors(30524) 
