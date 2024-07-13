import torch as tc

b1 = tc.rand([2, 2])
b2 = tc.rand([2, 2, 4])
b3 = tc.rand([4, 2, 2])
b4 = tc.rand([2, 2])
a1, r =tc.linalg.qr(b1)
b2_ = tc.einsum('ij, jkl->ikl', r, b2)
idx = b2_.shape
a2_, r = tc.linalg.qr(b2_.reshape([idx[0]*idx[1], idx[2]]))
a2 = a2_.reshape([idx[0], idx[1], -1])
b3_ = tc.einsum('ij, jkl->ikl', r, b3)
a4_h, r_h = tc.linalg.qr(b4.T.conj())
a4 = a4_h.T.conj()
r_r = r_h.T.conj()
b3_ = tc.einsum('ijk, kl->ijl', b3_, r_r)
# z = tc.einsum('ij, jkl, lmn, na->ikma', a1, a2, b3_, a4)

idx = b3_.shape
a3_h, r_h = tc.linalg.qr(b3_.reshape(idx[0], idx[1]*idx[2]).T.conj())
a3_ = a3_h.T.conj()
a3 = a3_.reshape([-1, idx[1], idx[2]])
r_r = r_h.T.conj()
c = tc.einsum('ij, jk->ik', r, r_r)
z = tc.einsum('ij, jkl, lb, bmn, na->ikma', a1, a2, r_r, a3, a4)
u, s_, v = tc.linalg.svd(r_r)
# a = tc.einsum('ij, jkl, lmn, na->ikma', b1, b2, b3, b4)

b = tc.einsum('ij, jkl, lmn, na->ikma', b1, b2, b3, b4)
print(tc.dist(z, b))

u1, s, v1 = tc.linalg.svd(b.reshape([4, 4]), full_matrices=False)
print(tc.dist(s, s_))
dc = 2
# dc = 2
# u_cut1 = u1[:, :dc].reshape([4, 2])
# s_cut1 = tc.diag(s[:dc])
# v_cut1 = v1[:dc, :].reshape([2, 4, 4])

# a_cut1 = u_cut1 @ s_cut1 @ v_cut1

