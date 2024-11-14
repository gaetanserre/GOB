import optimizers


bounds = optimizers.create_rect_bounds(-1, 1, 2)

prs = optimizers.PRS(bounds, 200_000)

f = lambda x: x.T @ x

print(prs.optimize(f))
