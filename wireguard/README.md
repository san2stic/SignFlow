# SignFlow WireGuard VPN Configuration

## Vue d'ensemble

Ce répertoire contient la configuration WireGuard pour l'accès distant sécurisé à l'infrastructure SignFlow.

## Topologie réseau

```
Internet
    |
    | Port 51820/UDP
    |
[WireGuard Server] (10.13.13.1)
    |
    | Réseau VPN: 10.13.13.0/24
    |
    +-- [Client 1] peer1 (10.13.13.2)
    +-- [Client 2] peer2 (10.13.13.3)
    +-- [Client 3] peer3 (10.13.13.4)
    +-- [Client 4] peer4 (10.13.13.5)
    +-- [Client 5] peer5 (10.13.13.6)
    |
    | Réseau Docker: 172.20.0.0/16
    |
    +-- Backend API     (172.20.0.x:8000)
    +-- Frontend        (172.20.0.x:80)
    +-- PostgreSQL      (172.20.0.x:5432)
    +-- MLflow UI       (172.20.0.x:5001)
    +-- Redis           (172.20.0.x:6379)
    +-- Elasticsearch   (172.20.0.x:9200)
```

## Services accessibles via VPN

Une fois connecté au VPN (10.13.13.0/24), tu peux accéder aux services via leurs noms Docker :

| Service | URL d'accès | Description |
|---------|-------------|-------------|
| **Frontend** | http://caddy | Interface web React |
| **Backend API** | http://backend:8000 | API FastAPI |
| **PostgreSQL** | postgresql://db:5432 | Base de données |
| **MLflow UI** | http://mlflow:5001 | Dashboard ML (avec profile mlops) |
| **Redis** | redis://redis:6379 | Cache |
| **Elasticsearch** | http://elasticsearch:9200 | Recherche |

## Configuration automatique

Le container WireGuard génère automatiquement :
- Les clés serveur (privée/publique)
- Les configs clients (peer1.conf, peer2.conf, etc.)
- Les QR codes pour mobile (peer1.png, peer2.png, etc.)

Tous ces fichiers seront créés dans `./wireguard/config/` au premier démarrage.

## Premier démarrage

1. **Vérifier ton IP publique** (optionnel) :
   ```bash
   curl -4 ifconfig.me
   ```

2. **Démarrer le stack avec WireGuard** :
   ```bash
   docker-compose -f docker-compose.prod.yml up -d wireguard
   ```

3. **Attendre la génération des configs** (~10 secondes) :
   ```bash
   docker logs signflow_wireguard
   ```

4. **Récupérer les configs clients** :
   ```bash
   ls -lh wireguard/config/peer*
   ```

## Connexion client

### Sur ordinateur (Linux/macOS/Windows)

1. Installer WireGuard : https://www.wireguard.com/install/

2. Copier la config client :
   ```bash
   # Exemple pour peer1
   cat wireguard/config/peer1/peer1.conf
   ```

3. Importer dans WireGuard et activer la connexion

### Sur mobile (iOS/Android)

1. Installer l'app WireGuard officielle

2. Scanner le QR code :
   ```bash
   # Ouvrir l'image QR générée (macOS)
   open wireguard/config/peer1/peer1.png

   # OU afficher un QR texte dans le terminal
   docker exec signflow_wireguard sh -lc 'qrencode -t ansiutf8 < /config/peer1/peer1.conf'
   ```

## Vérification de connexion

Une fois connecté :

```bash
# Ping le serveur WireGuard
ping 10.13.13.1

# Tester l'accès au backend
curl http://backend:8000/health

# Tester l'accès au frontend
curl http://caddy
```

## Sécurité

- ✅ Chiffrement WireGuard (ChaCha20Poly1305)
- ✅ Authentification par clés publiques/privées
- ✅ Pas d'accès Internet via VPN (ALLOWEDIPS=10.13.13.0/24)
- ✅ Isolation réseau Docker
- ⚠️ **Ne JAMAIS commit les clés privées dans git**

## Troubleshooting

### Le VPN ne se connecte pas

```bash
# Vérifier que le port UDP 51820 est ouvert sur ton firewall/routeur
sudo ufw allow 51820/udp  # Linux
# ou configurer le port forwarding sur ton routeur

# Vérifier les logs WireGuard
docker logs signflow_wireguard
```

### Le port 51820/UDP reste fermé depuis Internet

1. Définir `WG_SERVERURL` dans `.env` avec ton IPv4 publique ou un domaine.
2. Configurer un port-forward UDP `51820 -> 192.168.0.49:51820` sur le routeur (adapter l'IP LAN du serveur).
3. Vérifier que l'IP WAN du routeur correspond à `curl -4 ifconfig.me`.
4. Si l'IP WAN est différente, tu es probablement derrière CGNAT: demander une IPv4 publique à l'opérateur, ou passer par un VPS/relais.

Si l'import iOS/Android affiche "configuration WireGuard invalide", vérifie la ligne `Endpoint` dans `peerX.conf` :

```ini
Endpoint = [IPv6_DU_SERVEUR]:51820
```

Pour un endpoint IPv4 ou domaine, ne pas mettre de crochets.

### Les services Docker ne sont pas accessibles

```bash
# Vérifier que tous les services sont sur le réseau signflow_network
docker network inspect signflow_prod_signflow_network

# Redémarrer le stack complet
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml up -d
```

### Régénérer les configs clients

```bash
# Supprimer les configs existantes
rm -rf wireguard/config/*

# Redémarrer WireGuard (régénération auto)
docker-compose -f docker-compose.prod.yml restart wireguard
```

## Variables d'environnement

| Variable | Valeur | Description |
|----------|--------|-------------|
| `WG_SERVERURL` | *(vide)* | IP publique IPv4 ou domaine (recommandé). Injecté dans `SERVERURL` du conteneur. |
| `SERVERURL` | `auto` | Détection auto IP publique (fallback si `WG_SERVERURL` non défini) |
| `SERVERPORT` | `51820` | Port UDP WireGuard |
| `PEERS` | `5` | Nombre de clients |
| `INTERNAL_SUBNET` | `10.13.13.0` | Réseau VPN |
| `ALLOWEDIPS` | `10.13.13.0/24` | Trafic routé via VPN |

## Fichiers générés

```
wireguard/config/
├── server/
│   ├── privatekey
│   ├── publickey
│   └── wg0.conf
├── peer1/
│   ├── peer1.conf      # Config client 1
│   ├── peer1.png       # QR code client 1
│   ├── privatekey-peer1
│   └── publickey-peer1
├── peer2/
│   └── ...
└── peer5/
    └── ...
```

## Notes importantes

- Le container nécessite `NET_ADMIN` et `SYS_MODULE` capabilities
- Le module kernel `wireguard` doit être disponible sur l'hôte
- Les configs sont persistées dans `./wireguard/config/`
- Ajoute `wireguard/config/` dans `.gitignore` pour protéger les clés
