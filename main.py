from training import trainer


def main():
    # python main.py --config_dir training/configs/dof6plane_deina_rsd.json --taskname dof6_sse
    trainer.main()


if __name__ == "__main__":
    main()
