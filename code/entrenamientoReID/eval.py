import torchreid
from datasets.sportsreid_dataset import register_dataset

def main():
    # 1. Registra el dataset (esto habilita el nombre 'sportsreid')
    register_dataset()

    # 2. Carga datos
    datamanager = torchreid.data.ImageDataManager(
        root=r'C:\Users\Soriano\OneDrive\Documentos\entrenamientoReID\player_crops_TEST_GT_Bundesliga', # <--- AQUÍ va tu carpeta física
        sources='sportsreid',                   # <--- AQUÍ va el nombre de la lógica registrada
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=100,
        transforms=['random_flip'],
        use_gpu=True
    )

    # 3. Carga modelo
    # NOTA: num_classes da igual para testear, pero ponlo para que no falle la construcción
    model = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=datamanager.num_train_pids if datamanager.num_train_pids > 0 else 100, 
        loss='softmax',
        pretrained=False
    )

    # 4. Carga tus pesos entrenados
    torchreid.utils.load_pretrained_weights(
        model,
        r'C:\Users\Soriano\OneDrive\Documentos\entrenamientoReID\reid_checkpoints\model.osnet.pth.tar-10'
    )

    # 5. Crear el motor y lanzar evaluación
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=None
    )

    print("\nComenzando evaluación Rank-1 y mAP...")
    engine.test()

if __name__ == '__main__':
    main()