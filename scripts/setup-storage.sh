#!/usr/bin/env bash
# =============================================================================
# setup-storage.sh — Auto-configuration des disques pour MinIO (SignFlow)
# =============================================================================
# Détecte tous les disques vides disponibles, les formate en XFS et les monte
# dans /mnt/disk1, /mnt/disk2, etc. Génère minio-volumes.env à sourcer avant
# docker compose.
#
# Usage:
#   sudo bash scripts/setup-storage.sh
#
# Génère: minio-volumes.env (à la racine du projet, chemin relatif au script)
# =============================================================================

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
MOUNT_BASE="/mnt"
FS_TYPE="xfs"
MINIO_DATA_SUBDIR="minio-data"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_OUTPUT_FILE="${PROJECT_ROOT}/minio-volumes.env"

# ── Couleurs ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "\n${BLUE}═══ $* ${NC}"; }

# ── Vérifications pré-démarrage ───────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
    log_error "Ce script doit être exécuté avec sudo ou en tant que root."
    echo "  Usage: sudo bash scripts/setup-storage.sh"
    exit 1
fi

# Vérifier que les outils nécessaires sont présents
for tool in lsblk blkid mkfs.xfs mount; do
    if ! command -v "$tool" &>/dev/null; then
        log_error "Outil manquant : $tool"
        echo "  Installer avec : apt-get install -y util-linux xfsprogs"
        exit 1
    fi
done

# ── Identifier le disque système ─────────────────────────────────────────────
log_step "Détection du disque système"

ROOT_DEV=$(df / | tail -1 | awk '{print $1}')
# Récupérer le disque parent (ex: /dev/sda1 → sda, /dev/nvme0n1p1 → nvme0n1)
ROOT_DISK=$(lsblk -no PKNAME "$ROOT_DEV" 2>/dev/null | head -1)
if [[ -z "$ROOT_DISK" ]]; then
    # Fallback : enlever le suffixe numérique
    ROOT_DISK=$(echo "$ROOT_DEV" | sed 's|/dev/||; s|p[0-9]*$||; s|[0-9]*$||')
fi
log_info "Disque système détecté : /dev/${ROOT_DISK} (partition root: ${ROOT_DEV}) — sera ignoré"

# ── Scanner les disques candidats ─────────────────────────────────────────────
log_step "Scan des disques disponibles"

declare -a CANDIDATE_DISKS=()

while IFS= read -r line; do
    NAME=$(awk '{print $1}' <<< "$line")
    TYPE=$(awk '{print $2}' <<< "$line")
    FSTYPE=$(awk '{print $3}' <<< "$line")
    MOUNTPOINT=$(awk '{print $4}' <<< "$line")

    # Filtres successifs
    # 1. Seulement les disques entiers (pas les partitions, loops, etc.)
    [[ "$TYPE" != "disk" ]] && continue

    # 2. Exclure le disque système
    [[ "$NAME" == "$ROOT_DISK" ]] && continue

    # 3. Exclure si déjà monté
    if [[ -n "$MOUNTPOINT" && "$MOUNTPOINT" != " " ]]; then
        log_warn "Skip /dev/$NAME : déjà monté sur '$MOUNTPOINT'"
        continue
    fi

    # 4. Exclure si filesystem existant (données à préserver)
    if [[ -n "$FSTYPE" && "$FSTYPE" != " " ]]; then
        log_warn "Skip /dev/$NAME : filesystem existant ($FSTYPE) — données détectées, skip pour sécurité"
        continue
    fi

    # 5. Exclure si le disque a des partitions enfants
    PART_COUNT=$(lsblk -rno TYPE "/dev/$NAME" 2>/dev/null | grep -c "^part$" || true)
    if [[ "$PART_COUNT" -gt 0 ]]; then
        log_warn "Skip /dev/$NAME : $PART_COUNT partition(s) existante(s) détectée(s)"
        continue
    fi

    SIZE=$(lsblk -no SIZE "/dev/$NAME" 2>/dev/null | head -1 | tr -d ' ')
    log_info "Candidat trouvé : /dev/$NAME  (taille: $SIZE)"
    CANDIDATE_DISKS+=("/dev/$NAME")

done < <(lsblk -rno NAME,TYPE,FSTYPE,MOUNTPOINT 2>/dev/null)

# ── Cas : aucun disque disponible ────────────────────────────────────────────
if [[ ${#CANDIDATE_DISKS[@]} -eq 0 ]]; then
    log_warn "Aucun disque vide trouvé. MinIO utilisera le stockage système."
    FALLBACK_DIR="/opt/signflow-data/minio-data"
    mkdir -p "$FALLBACK_DIR"
    chmod 750 "$FALLBACK_DIR"

    cat > "$ENV_OUTPUT_FILE" <<EOF
# Généré par setup-storage.sh le $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# Mode: stockage système (aucun disque dédié détecté)
MINIO_VOLUMES=${FALLBACK_DIR}
MINIO_DISK_COUNT=1
MINIO_STORAGE_MODE=system
EOF

    log_info "Fichier généré : $ENV_OUTPUT_FILE"
    cat "$ENV_OUTPUT_FILE"
    echo ""
    log_warn "Performance limitée — MinIO s'exécutera sur le disque système."
    log_warn "Pour de meilleures performances, ajoutez des disques dédiés."
    exit 0
fi

# ── Afficher les candidats et demander confirmation ───────────────────────────
log_step "Disques qui seront formatés"

echo ""
echo "┌────────────────────────────────────────────────────────────┐"
echo "│  ATTENTION — OPÉRATION DESTRUCTIVE IRRÉVERSIBLE            │"
echo "├─────┬────────────┬────────────────────────────────────────┤"
printf "│ %-3s │ %-10s │ %-38s │\n" "N°" "Disque" "Taille"
echo "├─────┼────────────┼────────────────────────────────────────┤"

for i in "${!CANDIDATE_DISKS[@]}"; do
    DISK="${CANDIDATE_DISKS[$i]}"
    SIZE=$(lsblk -no SIZE "$DISK" 2>/dev/null | head -1 | tr -d ' ')
    printf "│ %-3s │ %-10s │ %-38s │\n" "$((i+1))" "$DISK" "$SIZE"
done

echo "└─────┴────────────┴────────────────────────────────────────┘"
echo ""
echo -e "${RED}⚠  TOUTES LES DONNÉES SUR CES DISQUES SERONT EFFACÉES DÉFINITIVEMENT.${NC}"
echo ""
read -rp "  Tapez exactement 'OUI' pour confirmer le formatage : " CONFIRM
echo ""

if [[ "$CONFIRM" != "OUI" ]]; then
    log_warn "Opération annulée par l'utilisateur."
    exit 0
fi

# ── Formatage, montage et fstab ───────────────────────────────────────────────
log_step "Formatage et montage des disques"

declare -a MOUNTED_PATHS=()
DISK_NUM=1

for DISK in "${CANDIDATE_DISKS[@]}"; do
    MOUNT_POINT="${MOUNT_BASE}/disk${DISK_NUM}"
    DATA_DIR="${MOUNT_POINT}/${MINIO_DATA_SUBDIR}"

    log_info "Formatage de $DISK en XFS (label: minio${DISK_NUM})..."
    mkfs.xfs -f -L "minio${DISK_NUM}" "$DISK" > /dev/null 2>&1
    log_info "  ✓ Formatage terminé"

    log_info "Montage sur $MOUNT_POINT..."
    mkdir -p "$MOUNT_POINT"

    # Récupérer l'UUID du disque fraîchement formaté
    UUID=$(blkid -s UUID -o value "$DISK")
    if [[ -z "$UUID" ]]; then
        log_error "Impossible de récupérer l'UUID de $DISK"
        exit 1
    fi

    # Supprimer toute entrée fstab existante pour ce disque ou ce point de montage
    sed -i "\|UUID=${UUID}|d" /etc/fstab
    sed -i "\|${MOUNT_POINT}|d" /etc/fstab

    # Ajouter la nouvelle entrée avec nofail (démarrage tolérant aux pannes disque)
    echo "UUID=${UUID}  ${MOUNT_POINT}  xfs  defaults,nofail  0  2" >> /etc/fstab
    log_info "  ✓ Entrée fstab ajoutée (UUID=${UUID}, nofail)"

    # Monter le disque
    mount "$MOUNT_POINT"
    log_info "  ✓ Monté sur $MOUNT_POINT"

    # Créer le répertoire de données MinIO
    mkdir -p "$DATA_DIR"
    chmod 750 "$DATA_DIR"
    log_info "  ✓ Répertoire MinIO créé : $DATA_DIR"

    MOUNTED_PATHS+=("$DATA_DIR")
    DISK_NUM=$((DISK_NUM + 1))
done

# ── Générer la variable MINIO_VOLUMES ─────────────────────────────────────────
log_step "Génération de la configuration MinIO"

COUNT=${#MOUNTED_PATHS[@]}
STORAGE_MODE=""

if [[ $COUNT -eq 1 ]]; then
    # 1 disque : chemin simple, mode single-node single-drive
    VOLUMES_STR="${MOUNTED_PATHS[0]}"
    STORAGE_MODE="single-drive"
    log_info "Mode : Single-drive (1 disque)"

elif [[ $COUNT -ge 4 ]]; then
    # 4+ disques : notation brace expansion MinIO pour erasure coding automatique
    # MinIO répartit les données et calcule des parités (perte jusqu'à N/2 disques tolérée)
    VOLUMES_STR="${MOUNT_BASE}/disk{1...${COUNT}}/${MINIO_DATA_SUBDIR}"
    STORAGE_MODE="erasure-coding"
    log_info "Mode : Erasure coding ($COUNT disques) — tolérance jusqu'à $((COUNT/2)) pannes"

else
    # 2-3 disques : JBOD (liste de chemins séparés par espace)
    VOLUMES_STR="${MOUNTED_PATHS[*]}"
    STORAGE_MODE="jbod"
    log_info "Mode : JBOD ($COUNT disques) — sans redondance automatique"
    log_warn "Tip : 4+ disques activent le erasure coding (meilleure tolérance aux pannes)"
fi

# ── Générer le fichier minio-volumes.env ─────────────────────────────────────
cat > "$ENV_OUTPUT_FILE" <<EOF
# Généré automatiquement par setup-storage.sh
# Date : $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# Disques configurés : $COUNT
# Mode MinIO : $STORAGE_MODE
#
# UTILISATION :
#   source minio-volumes.env
#   docker compose -f docker-compose.yml -f docker-compose.prod.yml \\
#     -f docker-compose.server.yml --env-file .env.server up -d

MINIO_VOLUMES=${VOLUMES_STR}
MINIO_DISK_COUNT=${COUNT}
MINIO_STORAGE_MODE=${STORAGE_MODE}
EOF

log_info "Fichier généré : $ENV_OUTPUT_FILE"

# ── Afficher le résumé ────────────────────────────────────────────────────────
log_step "Résumé"

echo ""
echo "  Disques configurés : $COUNT"
echo "  Mode MinIO : $STORAGE_MODE"
echo "  MINIO_VOLUMES = $VOLUMES_STR"
echo ""
echo "  Points de montage :"
for i in "${!MOUNTED_PATHS[@]}"; do
    USAGE=$(df -h "${MOUNTED_PATHS[$i]}" | tail -1 | awk '{print $2 " total, " $4 " disponible"}')
    echo "    ${MOUNTED_PATHS[$i]}  ($USAGE)"
done
echo ""

# Générer aussi les lignes de volumes Docker pour docker-compose.server.yml
echo "  ── Bind mounts à ajouter dans docker-compose.server.yml ──"
echo "  (section volumes du service minio)"
for i in "${!MOUNTED_PATHS[@]}"; do
    echo "    - ${MOUNTED_PATHS[$i]}:${MOUNTED_PATHS[$i]}"
done
echo ""
log_info "Prêt ! Lancer le déploiement avec :"
echo ""
echo "  source minio-volumes.env"
echo "  docker compose \\"
echo "    -f docker-compose.yml \\"
echo "    -f docker-compose.prod.yml \\"
echo "    -f docker-compose.server.yml \\"
echo "    --env-file .env.server \\"
echo "    up -d"
echo ""
