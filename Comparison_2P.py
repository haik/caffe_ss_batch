"""
1. x = x1 + (-x2)
2. [r], [r0]...[rl]
3. [c] = [x] - [r] => c
4. Bitwise addition => x = c + r
5. MSB(x) is 0 or 1?
"""
import random
import math
import time

def binaryconvert(number):
	number = bin8(number)				# binary expression
	number = number[::-1]
	binaryform = [0] * n
	for i in range(n):
		binaryform[i] = int(number[i])
	return binaryform

def BitReshare(u1, u2, u3):
	r12 = random.randint(0, 1)    	# S1 -> S2
	r23 = random.randint(0, 1)		# S2 -> S3
	r31 = random.randint(0, 1)		# S3 -> S1
	w1 = (u1 + r12 - r31) % 2
	w2 = (u2 + r23 - r12) % 2
	w3 = (u3 + r31 - r23) % 2
	return w1, w2, w3

def BitMultiply(u1, u2, v1, v2):
	a1 = random.randint(0, 1)			# S1 <- triplet
	b1 = random.randint(0, 1)
	c1 = random.randint(0, 1)

	a2 = random.randint(0, 1)			# S2 <- triplet
	b2 = random.randint(0, 1)
	c2 = ((a1 + a2) * (b1 + b2) - c1) % 2

	u_a = (u1 - a1 + u2 - a2) % 2			# S1 <-> S2
	v_b = (v1 - b1 + v2 - b2) % 2

	z1 = (c1 + u_a * b1 + v_b * a1)	% 2				# S1
	z2 = (c2 + u_a * b2 + v_b * a2 + u_a * v_b) % 2		# S2

	return z1, z2					# z1 + z2 = xy

def BitAddition(u1, u2, v1, v2):			# input 10 based numbers
	krange = int(math.log(n, 2))
	
	w1 = [0] * n
	w2 = [0] * n
	# w = [0] * n

	p1 = [0] * (n*2)
	p2 = [0] * (n*2)

	s1 = [0] * (n*2)
	s2 = [0] * (n*2)

	for j in range(n):
		p1[j] = u1[j] ^ v1[j]
		p2[j] = u2[j] ^ v2[j]
		s1[j], s2[j] = BitMultiply(u1[j], u2[j], v1[j], v2[j])

	k_mul = 0   # 统计乘法次数
	for k in range(krange):
		lrange = 2**k
		mrange = n // (lrange+1)
		for l in range(lrange):
			for m in range(mrange):
				pos1 = lrange + l + lrange*2*m
				pos2 = lrange - 1 + lrange*2*m
				if pos1 < n and pos2 < n:
					k_mul += 1
					temp1, temp2 = BitMultiply(p1[pos1], p2[pos1], s1[pos2], s2[pos2])
					s1[pos1] = s1[pos1] ^ temp1
					s2[pos1] = s2[pos1] ^ temp2
					p1[pos1], p2[pos1] = BitMultiply(p1[pos1], p2[pos1], p1[pos2], p2[pos2])
	print(k_mul)
	w1[0] = u1[0] ^ v1[0]
	w2[0] = u2[0] ^ v2[0]
	for j in range(1, n):
		w1[j] = u1[j] ^ v1[j] ^ s1[j-1]
		w2[j] = u2[j] ^ v2[j] ^ s2[j-1]
	return w1, w2

def BitExtraction(u1, u2):
	r1 = random.randint(0, 2**(n-1))
	r2 = random.randint(0, 2**(n-1))
	rb = binaryconvert(r1+r2)
	q1 = [0] * n
	q2 = [0] * n
	for j in range(n):
		q1[j] = random.randint(0, 1)
		q2[j] = rb[j] ^ q1[j]
	''' S1 -> S2: r2, q2
		S1 -> S3: r3, q3 '''
	v1 = u1 - r1
	v2 = u2 - r2
	v = v1 + v2
	vb = binaryconvert(v)
	v1 = [0] * n
	v2 = [0] * n
	for j in range(n):
		# v1[j] = random.randint(0, 1)
		v1[j] = random.randint(0, 1)
		v2[j] = vb[j] ^ v1[j]
	w1, w2 = BitAddition(v1, v2, q1, q2)
	return w1, w2

if __name__ == '__main__':
	n = 32
	bin8 = lambda x : ''.join(reversed( [str((x >> i) & 1) for i in range(n)] ) )

	a = -8
	b = -12
	################ Test BitAddition ##############
	t1 = time.time()
	a = bin8(a)				# binary expression
	a = a[::-1]				# in reverse order
	b = bin8(b)
	b = b[::-1]

	u = [0] * n
	u1 = [0] * n
	u2 = [0] * n
	# u3 = [0] * n
	
	v = [0] * n
	v1 = [0] * n
	v2 = [0] * n
	# v3 = [0] * n

	for j in range(n):
		u[j] = int(a[j])
		u1[j] = random.randint(0, 1)
		# u2[j] = random.randint(0, 1)
		u2[j] = u[j] ^ u1[j]

		v[j] = int(b[j])
		v1[j] = random.randint(0, 1)
		# v2[j] = random.randint(0, 1)
		v2[j] = v[j] ^ v1[j]

	c1, c2 = BitAddition(u1, u2, v1, v2)
	t2 = time.time()
	###################################################
	c = [0] * n
	for j in range(n):
		c[j] = c1[j] ^ c2[j]
	print(c)
	print(t2-t1)

	# v = BitReshare(1, 1, 1)
	# print(v)

	# w = BitMultiply(1, 0, 0, 1, 0, 0)
	# print(w)

	t3 = time.time()
	bits1, bits2 = BitExtraction(-120, -34)
	t4 = time.time()
	z = [0] * n
	for j in range(n):
		z[j] = bits1[j] ^ bits2[j]
	print(z)
	print(t4-t3)