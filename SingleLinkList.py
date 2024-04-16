# 用python做线性表

import sys

sys.setrecursionlimit(3000)
size = 0


class SingleLinkList:
    def __init__(self, x=None):
        self.data = x
        self.next = None


def add(struct, x):
    if struct.next is not None:
        add(struct.next, x)
    else:
        struct.next = SingleLinkList(x)


def delete(struct, x):
    if struct.next is not None:
        if struct.next.data == x:
            print('found {} to delete.'.format(struct.next.data))
            if struct.next.next is not None:
                struct.next = struct.next.next
            else:
                struct.next = None
        else:
            delete(struct.next, x)
    else:
        print('not found.')


def output(struct):
    if struct.data is None:
        output(struct.next)
    else:
        print(struct.data)
        if struct.next is not None:
            output(struct.next)
        else:
            print('tail.')


def search(struct, x):
    if struct.data is None:
        search(struct.next, x)
    else:
        if struct.data == x:
            print(struct.data)
        else:
            if struct.next is not None:
                search(struct.next, x)
            else:
                print('not found')


def search_index(struct, n, index=0):
    if index == n:
        print('found. ', struct.data)
    else:
        if struct.next is not None:
            search_index(struct.next, n, index+1)
        else:
            print('out of index.')


def insert(struct, n, x, index=0):
    if struct.next is not None:
        if index == n:
            print('index: ', index)
            cache_node = SingleLinkList(x)
            cache_node.next = struct.next
            struct.next = cache_node
            print('Insert complete. ')
            return 0
        else:
            insert(struct.next, n, x, index+1)
    else:
        if index == n:
            print('index: ', index)
            cache_node = SingleLinkList(x)
            struct.next = cache_node
            print('Insert complete. ')
            return 0
        else:
            print('out of index.')


def get_length(struct, length=0):
    if struct.next is None:
        print('length: ', length)
    else:
        get_length(struct.next, length+1)


if __name__ == '__main__':
    line = SingleLinkList()
    for i in range(20):
        add(line, i)
    search(line, 15)
    delete(line, 49)
    insert(line, 20, 0.55)
    output(line)
    search_index(line, 15)
    search_index(line, 21)
    get_length(line)


