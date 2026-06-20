export type Comparator<T> = (a: T, b: T) => number;

interface AVLNode<T> {
    value: T;
    left:  AVLNode<T> | null;
    right: AVLNode<T> | null;
    height: number;
}

function height<T>(n: AVLNode<T> | null): number {
    return n === null ? 0 : n.height;
}

function updateHeight<T>(n: AVLNode<T>): void {
    n.height = 1 + Math.max(height(n.left), height(n.right));
}

function rotateRight<T>(y: AVLNode<T>): AVLNode<T> {
    const x = y.left!;
    y.left  = x.right;
    x.right = y;
    updateHeight(y);
    updateHeight(x);
    return x;
}

function rotateLeft<T>(x: AVLNode<T>): AVLNode<T> {
    const y = x.right!;
    x.right = y.left;
    y.left  = x;
    updateHeight(x);
    updateHeight(y);
    return y;
}

function rebalance<T>(n: AVLNode<T>): AVLNode<T> {
    updateHeight(n);
    const bf = height(n.right) - height(n.left);
    if (bf > 1) {
        if (height(n.right!.right) < height(n.right!.left))
            n.right = rotateRight(n.right!);
        return rotateLeft(n);
    }
    if (bf < -1) {
        if (height(n.left!.left) < height(n.left!.right))
            n.left = rotateLeft(n.left!);
        return rotateRight(n);
    }
    return n;
}

function insertNode<T>(n: AVLNode<T> | null, value: T, cmp: Comparator<T>): AVLNode<T> {
    if (n === null) return { value, left: null, right: null, height: 1 };
    const c = cmp(value, n.value);
    if      (c < 0) n.left  = insertNode(n.left,  value, cmp);
    else if (c > 0) n.right = insertNode(n.right, value, cmp);
    return rebalance(n);
}

function hasNode<T>(n: AVLNode<T> | null, value: T, cmp: Comparator<T>): boolean {
    if (n === null) return false;
    const c = cmp(value, n.value);
    if (c < 0) return hasNode(n.left,  value, cmp);
    if (c > 0) return hasNode(n.right, value, cmp);
    return true;
}

function minNode<T>(n: AVLNode<T>): AVLNode<T> {
    return n.left === null ? n : minNode(n.left);
}

function removeNode<T>(n: AVLNode<T> | null, value: T, cmp: Comparator<T>): AVLNode<T> | null {
    if (n === null) return null;
    const c = cmp(value, n.value);
    if (c < 0) {
        n.left  = removeNode(n.left,  value, cmp);
    } else if (c > 0) {
        n.right = removeNode(n.right, value, cmp);
    } else {
        if (n.left  === null) return n.right;
        if (n.right === null) return n.left;
        const succ  = minNode(n.right);
        n.value     = succ.value;
        n.right     = removeNode(n.right, succ.value, cmp);
    }
    return rebalance(n);
}

function cloneNode<T>(n: AVLNode<T> | null, copyValue: (v: T) => T): AVLNode<T> | null {
    if (n === null) return null;
    return {
        value:  copyValue(n.value),
        left:   cloneNode(n.left,  copyValue),
        right:  cloneNode(n.right, copyValue),
        height: n.height,
    };
}

export class AVLTree<T> {
    private root: AVLNode<T> | null = null;
    private readonly cmp: Comparator<T>;
    size = 0;

    constructor(cmp: Comparator<T>) {
        this.cmp = cmp;
    }

    insert(value: T): void {
        this.root = insertNode(this.root, value, this.cmp);
        this.size++;
    }

    // cmp override is used by BoardState.sortedHistory (boardState.ts) to query
    // with cmpNonStrict while the tree is ordered by cmpStrict — see the comment
    // above sortedHistory in the BoardState constructor for the full rationale.
    has(value: T, cmp?: Comparator<T>): boolean {
        return hasNode(this.root, value, cmp ?? this.cmp);
    }

    remove(value: T): void {
        this.root = removeNode(this.root, value, this.cmp);
        this.size--;
    }

    clone(copyValue: (v: T) => T): AVLTree<T> {
        const t  = new AVLTree<T>(this.cmp);
        t.root   = cloneNode(this.root, copyValue);
        t.size   = this.size;
        return t;
    }
}
