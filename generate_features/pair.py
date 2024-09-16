import click
import torch

ATOM_N, ATOM_CA, ATOM_C, ATOM_O, ATOM_CB = 0, 1, 2, 3, 4


def NormVec(V):
    eps = 1e-10
    axis_x = V[:, 2] - V[:, 1]
    axis_x /= torch.norm(axis_x, dim=-1).unsqueeze(1) + eps
    axis_y = V[:, 0] - V[:, 1]
    axis_z = torch.cross(axis_x, axis_y, dim=1)
    axis_z /= torch.norm(axis_z, dim=-1).unsqueeze(1) + eps
    axis_y = torch.cross(axis_z, axis_x, dim=1)
    axis_y /= torch.norm(axis_y, dim=-1).unsqueeze(1) + eps
    Vec = torch.stack([axis_x, axis_y, axis_z], dim=1)
    return Vec


def QuaternionMM(q1, q2):
    a = q1[..., 0] * q2[..., 0] - (q1[..., 1:] * q2[..., 1:]).sum(-1)
    bcd = torch.cross(q2[..., 1:], q1[..., 1:], dim=-1) + q1[..., 0].unsqueeze(-1) * q2[..., 1:] + q2[..., 0].unsqueeze(-1) * q1[..., 1:]
    q = torch.cat([a.unsqueeze(-1), bcd], dim=-1)
    return q


def NormQuaternionMM(q1, q2):
    q = QuaternionMM(q1, q2)
    return q / torch.sqrt((q * q).sum(-1, keepdim=True))


def Rotation2Quaternion(r):
    a = torch.sqrt(r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2] + 1) / 2.0
    b = (r[..., 2, 1] - r[..., 1, 2]) / (4 * a)
    c = (r[..., 0, 2] - r[..., 2, 0]) / (4 * a)
    d = (r[..., 1, 0] - r[..., 0, 1]) / (4 * a)
    q = torch.stack([a, b, c, d], dim=-1)
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True))
    return q


def NormQuaternion(q):
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True))
    q = torch.sign(torch.sign(q[..., 0]) + 0.5).unsqueeze(-1) * q
    return q


@click.command()
@click.option("--coordinate_file", required=True, type=str)
@click.option("--saved_folder", required=True, type=str)
def main(coordinate_file, saved_folder):
    pos14 = torch.load(coordinate_file)["pos14"]
    assert pos14.shape[1:] == (14, 3)
    L = pos14.shape[0]

    # N CA C
    rotation = NormVec(pos14[:, :3, :])
    U, _, V = torch.svd(torch.eye(3).unsqueeze(0).permute(0, 2, 1) @ rotation)
    d = torch.sign(torch.det(U @ V.permute(0, 2, 1)))
    Id = torch.eye(3).repeat(L, 1, 1)
    Id[:, 2, 2] = d
    r = V @ (Id @ U.permute(0, 2, 1))
    q = Rotation2Quaternion(r)
    q_1 = torch.cat([q[..., 0].unsqueeze(-1), -q[..., 1:]], dim=-1)
    QAll = NormQuaternionMM(q.unsqueeze(1).repeat(1, L, 1), q_1.unsqueeze(0).repeat(L, 1, 1))

    QAll[..., 0][torch.isnan(QAll[..., 0])] = 1.0
    QAll[torch.isnan(QAll)] = 0.0
    QAll = NormQuaternion(QAll)

    # QAll = [L, L, 4]

    xyz_CA = torch.einsum("a b i, a i j -> a b j", pos14[:, ATOM_CA].unsqueeze(0) - pos14[:, ATOM_CA].unsqueeze(1), r)
    # xyz_C = torch.einsum('a b i, a i j -> a b j', pos14[:, ATOM_C].unsqueeze(0) - pos14[:, ATOM_CA].unsqueeze(1), r)
    # xyz_N = torch.einsum('a b i, a i j -> a b j', pos14[:, ATOM_N].unsqueeze(0) - pos14[:, ATOM_CA].unsqueeze(1), r)

    # xyz_CA = [L, L, 3]

    torch.save(torch.cat([xyz_CA, QAll], dim=-1).detach().cpu().clone(), f"{saved_folder}/pair.pt")


if __name__ == "__main__":
    main()
