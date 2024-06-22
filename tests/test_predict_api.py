import unittest
import os
import json
from app.failure_predict import app, scaler, model
import pandas as pd
import joblib

class PredictAPITestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

        base_dir = os.path.dirname(os.path.abspath(__file__))
        scaler_path = os.path.join(base_dir, '..', 'data', 'scaler.pkl')
        model_path = os.path.join(base_dir, '..', 'data', 'model.pkl')
        
        # Chargement du scaler et du modèle
        self.scaler = joblib.load(scaler_path)
        self.model = joblib.load(scaler_model)

    
    def test_predict(self):
        # Exemple de données de test valides / source : df_test.drop(columns='SK_ID_CURR').loc[df_test.index == 41868].values.to_list()
        data = {
            "data": [[1.0, 0.0, 0.0, 0.0, 135000.0, 104256.0, 10683.0, 0.0246124267578125, -20411.0, -2384.144673644465, -11192.0, -1409.0, 12.061120673168864, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 2.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5021288766017784, 0.69873046875, 0.556640625, 0.0927734375, 0.08697509765625, 0.98291015625, 0.76904296875, 0.0419921875, 0.0, 0.2069091796875, 0.166748046875, 0.208251953125, 0.11480712890625, 0.07562255859375, 0.0872802734375, 0.0, 0.0, 0.094482421875, 0.09027099609375, 0.98291015625, 0.042388916015625, 0.0, 0.2069091796875, 0.11737060546875, 0.08258056640625, 0.09088134765625, 0.0, 0.0, 0.068603515625, 2.0, 1.0, 1.0, -636.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.1568596911223838, 1.294921875, 135000.0, 0.07916259765625, 0.10247802734375, -606.0, -273.0, -424.25, 31122.25, -241.0, 1553.0, 566.0, -221.0, 0.0, 0.0, 0.0, 720000.0, 271940.625, 1087762.5, 662863.5, 165930.75, 663723.0, 0.0, 0.0, 0.0, 19890.0, 9945.0, 0.0, 0.0, 14.5, 58.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.29296875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.33740234375, 0.369384765625, 0.0, -273.0, -273.0, -273.0, 0.0, 951.0, 1553.0, 1252.0, -89.0, 0.0, 0.0, 2529.0669499454825, 720000.0, 382500.0, 765000.0, 331861.5, 0.0, 0.0, 0.0, 19890.0, 19890.0, 0.0, 0.0, 10.0, 20.0, -606.0, -545.0, -575.5, 1860.5, -241.0, 1.0, -120.0, -353.0, 0.0, 0.0, 0.0, 161752.5, 161381.25, 322762.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 19.0, 38.0, 8086.5, 8086.5, 8086.5, 44509.5, 44509.5, 44509.5, 44509.5, 44509.5, 1.0, 1.0, 1.0, 0.0727581558526331, 0.0, 0.0, 0.0, 44509.5, 44509.5, 11.0, 11.0, 11.0, 0.0, 0.0, 0.0, -636.0, -636.0, -636.0, 6.0, 6.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8086.5, 8086.5, 8086.5, 44509.5, 44509.5, 44509.5, 44509.5, 44509.5, 1.0, 1.0, 1.0, 0.0813350401030407, 0.0, 0.0, 0.0, 44509.5, 11.0, 11.0, 11.0, 0.0, 0.0, 0.0, -636.0, -636.0, 6.0, 6.0, 15335.495005212391, 23026.07569438685, 18992.577902574896, 213832.6497750664, 388400.3681637509, 295891.1003624827, 0.8680136968625471, 0.9724960004451992, 0.9219771929795116, 0.0495065132826205, 5391.166719112633, 8398.701763928935, 6758.399465697065, 256229.77447359124, 11.603878751620302, 13.271343104995513, 12.419537306654451, 0.0511846387857162, 0.0734696022340096, 0.0615860081607363, -983.9393558679828, -728.6436135207897, -847.7661293576753, 20.222093440882123, 48.71538538239107, -17.0, -19.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.7998046875, 0.0, 0.0, 0.0, 0.199951171875, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 8.0, 6.0, 30.0, 2.782415136338342, 1.927620751056112, 0.6136981844902039, 0.0, -7200.0, -36000.0, 27829366.0, 14413.5, 9294.24609375, 46471.23046875, 10471.23046875, 16494.24609375, -551.0, -587.0, -2935.0, 5.0, -37.58450031643749, -1.4979805534779356, 10807.89970831731, 144501.30663182566, 71459.92696452411, 2194196.075010313, 5774771783.284015, 150870.20884874288, 249709.1191530982, 208260.5792548088, 5681247.177952937, 6161559975.401364, 643.2251989941758, 106144.56254362108, 12291.586141020456, 181772.17957411736, 1646746455.926673, 588.6853645895891, 99849.99129008267, 13913.475108606524, 278357.2799127908, 1861134512.0830405, 5.60107948969578, 9868.221816684172, 611.3762542758341, 9006.376699061992, 166585632.44066778, 606.1951616913032, 43479.99321955231, 8304.015510556948, 87024.40035340004, 726435669.5414225, 246.490141117734, 7245.025963619086, 3606.65239276108, 122754.11534014322, 16746486.618475026, 2138.9476847993, 118779.53630811826, 18028.67616253976, 307243.45196188305, 2393936786.2001863, 222.8308038218557, 10427.538941127395, 0.029375204448806, 3.995894667975139, 0.5662353620444944, 9.505229848685346, 2.040582564561685, 0.1009723261032161, 6.803624647603705, 1.5416455632255812, 26.0430124848973, 15.130825141106506, 9.813542688910696e-05, 0.1160942100098135, 0.0072546956739591, 0.1505206835049767, 0.0110880543910541, 0.1649656526005888, 6.409519136408243, 1.7500525111734069, 16.38726195270698, 0.8333122375007191, 16.28755537656061, 10.20232974292151, 719.1568839537426, 66.30972775227627, 0.0, 16.953535469765836, 4.209069244804402, 0.7800356711351476, 0.1268797825562598, 0.8309993671250215, 0.9992290432081008, 0.9669076123190148, 0.0223835759895074, 0.0, 5.753408894770152e-05, 0.0003106840803175, 0.1220643231114435, 0.0291778745300633, 1.151038490305506, 0.0194394542512848, 0.0, 0.0001726022668431, 0.0001055840429204, 2.974820958070601e-05, 0.0, 0.0001610954490535, 0.0, 0.0052931361831885, 0.0003682181692652, 0.0493757551349174, 0.0037378135467383, 0.114366262010241, 0.0030692270769801, 0.0, 0.0, 0.0, 0.0, 0.0]]
        }
        # Envoi d'une requête POST à l'API
        response = self.app.post('/predict', data=json.dumps(data), content_type='application/json')
        # Le statut doit être 200
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        # La réponse doit être une liste
        self.assertIsInstance(response_data, list)
        # La liste doit contenir un seul item
        self.assertEqual(len(response_data), 1)
        # Les items doivent être de type float
        self.assertTrue(all(isinstance(x, float) for x in response_data))

    def test_predict_invalid_data(self):
        # Exemple de données invalides
        data = {
            "data": "invalid_data"
        }
        # Envoi d'une requête POST à l'API
        response = self.app.post('/predict', data=json.dumps(data), content_type='application/json')
        self.assertEqual(response.status_code, 400)

    def test_predict_missing_data(self):
        # Requête sans données
        response = self.app.post('/predict', data=json.dumps({}), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        
    def test_predict_non_json(self):
        # Requête sans contenu JSON
        response = self.app.post('/predict', data="some random text", content_type='text/plain')
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()
