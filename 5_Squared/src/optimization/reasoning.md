# Graph Theory & Portfolio Construction

## 1. Graph Theory → Clustering

Converts the correlation matrix into a tree structure using hierarchical clustering. Stocks that move together get grouped into the same branch.

```
          ──────────────────
         │                  │
      ───────            ───────
     │       │          │       │
   RTX    MCHP        USB     FITB
   DHI    JBHT        RF      AXP
   
   (cyclicals)        (financials)
```

## 2. Linear Algebra → Quasi-Diagonalization

Reorders the covariance matrix so correlated stocks sit next to each other. This makes the matrix block-diagonal, which is much more stable to work with than a random ordering.

```
Before reorder        After reorder
■ · · ■ · ·          ■ ■ · · · ·
· ■ · · · ■          ■ ■ · · · ·
· · ■ · ■ ·    →     · · ■ ■ · ·
■ · · ■ · ·          · · ■ ■ · ·
· · ■ · ■ ·          · · · · ■ ■
· ■ · · · ■          · · · · ■ ■
```

## 3. Recursive Bisection → Weight Allocation

Splits the tree in half repeatedly, allocating risk inversely proportional to variance at each split.

```
All capital = 100%
        ↓
Left cluster = 60%    Right cluster = 40%
(lower variance)      (higher variance)
        ↓                    ↓
  split again           split again
  30%    30%            25%    15%
```