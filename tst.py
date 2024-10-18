def main():
    import torch
    from torch import nn

    for shape in [[2], [2, 3]]:
        d = 3
        x = torch.randn([*shape, 10, d])
        y = torch.randn([*shape, d, 2])
        z = torch.bmm(x, y)
    pass


if __name__ == "__main__":
    main()
