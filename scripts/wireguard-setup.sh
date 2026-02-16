#!/bin/bash

# SignFlow WireGuard VPN Setup Script
# Ce script facilite le déploiement et la gestion du VPN WireGuard

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WG_CONFIG_DIR="$PROJECT_DIR/wireguard/config"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.prod.yml"

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonctions utilitaires
print_header() {
    echo -e "${BLUE}═══════════════════════════════════════${NC}"
    echo -e "${BLUE}  SignFlow WireGuard VPN Setup${NC}"
    echo -e "${BLUE}═══════════════════════════════════════${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Vérification des prérequis
check_requirements() {
    print_info "Vérification des prérequis..."

    # Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker n'est pas installé"
        exit 1
    fi
    print_success "Docker installé"

    # Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose n'est pas installé"
        exit 1
    fi
    print_success "Docker Compose installé"

    # Module WireGuard kernel (optionnel, car linuxserver/wireguard peut le gérer)
    if lsmod | grep -q wireguard; then
        print_success "Module kernel WireGuard chargé"
    else
        print_warning "Module kernel WireGuard non chargé (sera géré par le container)"
    fi

    echo ""
}

# Démarrage du VPN
start_vpn() {
    print_info "Démarrage du serveur WireGuard..."

    cd "$PROJECT_DIR"
    docker-compose -f "$COMPOSE_FILE" up -d wireguard

    print_info "Attente de l'initialisation (10 secondes)..."
    sleep 10

    if docker ps | grep -q signflow_wireguard; then
        print_success "WireGuard démarré avec succès"

        # Afficher les logs
        echo ""
        print_info "Derniers logs :"
        docker logs --tail 20 signflow_wireguard

        echo ""
        print_success "Configuration terminée !"
        print_info "Les fichiers de configuration sont dans : $WG_CONFIG_DIR"
    else
        print_error "Échec du démarrage de WireGuard"
        docker logs signflow_wireguard
        exit 1
    fi
}

# Arrêt du VPN
stop_vpn() {
    print_info "Arrêt du serveur WireGuard..."

    cd "$PROJECT_DIR"
    docker-compose -f "$COMPOSE_FILE" stop wireguard

    print_success "WireGuard arrêté"
}

# Afficher les configs clients
show_clients() {
    print_info "Configurations clients disponibles :\n"

    if [ ! -d "$WG_CONFIG_DIR" ]; then
        print_error "Aucune configuration trouvée. Démarre d'abord le VPN."
        exit 1
    fi

    local peer_count=0
    for peer_dir in "$WG_CONFIG_DIR"/peer*/; do
        if [ -d "$peer_dir" ]; then
            peer_count=$((peer_count + 1))
            peer_name=$(basename "$peer_dir")
            peer_conf="$peer_dir/${peer_name}.conf"
            peer_qr="$peer_dir/${peer_name}.png"

            echo -e "${GREEN}━━━ $peer_name ━━━${NC}"

            if [ -f "$peer_conf" ]; then
                echo "📄 Config : $peer_conf"
            fi

            if [ -f "$peer_qr" ]; then
                echo "📱 QR Code : $peer_qr"
            fi

            echo ""
        fi
    done

    if [ $peer_count -eq 0 ]; then
        print_warning "Aucun client configuré"
    else
        print_success "$peer_count clients configurés"
    fi
}

# Afficher la config d'un client spécifique
show_client_config() {
    local peer_name="$1"

    if [ -z "$peer_name" ]; then
        print_error "Usage: $0 show-config <peer_name>"
        echo "Exemple: $0 show-config peer1"
        exit 1
    fi

    local peer_conf="$WG_CONFIG_DIR/$peer_name/${peer_name}.conf"

    if [ ! -f "$peer_conf" ]; then
        print_error "Configuration '$peer_name' introuvable"
        exit 1
    fi

    echo -e "${GREEN}━━━ Configuration $peer_name ━━━${NC}\n"
    cat "$peer_conf"
    echo ""

    print_info "Pour se connecter :"
    echo "1. Copie cette config dans un fichier .conf"
    echo "2. Importe-la dans l'app WireGuard"
    echo "   - Desktop : wg-quick up /chemin/vers/$peer_name.conf"
    echo "   - Mobile : Scanne le QR code dans $WG_CONFIG_DIR/$peer_name/${peer_name}.png"
}

# Afficher un QR code dans le terminal
show_qr_code() {
    local peer_name="$1"

    if [ -z "$peer_name" ]; then
        print_error "Usage: $0 qr <peer_name>"
        echo "Exemple: $0 qr peer1"
        exit 1
    fi

    local peer_qr="$WG_CONFIG_DIR/$peer_name/${peer_name}.png"

    if [ ! -f "$peer_qr" ]; then
        print_error "QR code '$peer_name' introuvable"
        exit 1
    fi

    # Vérifier si qrencode est installé pour affichage ASCII
    if command -v qrencode &> /dev/null; then
        local peer_conf="$WG_CONFIG_DIR/$peer_name/${peer_name}.conf"
        echo -e "${GREEN}━━━ QR Code $peer_name (ASCII) ━━━${NC}\n"
        qrencode -t ansiutf8 < "$peer_conf"
        echo ""
    else
        print_info "QR Code PNG : $peer_qr"
        print_warning "Installe 'qrencode' pour afficher le QR code dans le terminal"
        echo "  macOS: brew install qrencode"
        echo "  Linux: sudo apt install qrencode"
    fi
}

# Régénérer les configs
regenerate_configs() {
    print_warning "⚠️  ATTENTION : Cela va supprimer TOUTES les configs existantes"
    read -p "Confirmer la régénération ? (y/N) " -n 1 -r
    echo

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Annulé"
        exit 0
    fi

    print_info "Suppression des configs existantes..."
    rm -rf "$WG_CONFIG_DIR"/*

    print_info "Redémarrage de WireGuard..."
    cd "$PROJECT_DIR"
    docker-compose -f "$COMPOSE_FILE" restart wireguard

    print_info "Attente de la régénération (10 secondes)..."
    sleep 10

    print_success "Configs régénérées"
    show_clients
}

# Afficher le statut
show_status() {
    print_info "État du VPN WireGuard :\n"

    if docker ps --format '{{.Names}}' | grep -q signflow_wireguard; then
        print_success "Container WireGuard : ✅ Running"

        # Stats du container
        echo ""
        docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" signflow_wireguard

        # Connexions actives (via wg show dans le container)
        echo ""
        print_info "Connexions actives :"
        docker exec signflow_wireguard wg show || print_warning "Impossible de récupérer les stats WireGuard"
    else
        print_error "Container WireGuard : ❌ Stopped"
    fi

    echo ""
    print_info "Port UDP : 51820"
    print_info "Réseau VPN : 10.13.13.0/24"
    print_info "Config dir : $WG_CONFIG_DIR"
}

# Afficher l'aide
show_help() {
    cat << EOF
Usage: $0 <command> [options]

Commands:
  start              Démarre le serveur WireGuard
  stop               Arrête le serveur WireGuard
  restart            Redémarre le serveur WireGuard
  status             Affiche l'état du VPN
  list               Liste tous les clients configurés
  show-config <peer> Affiche la config d'un client (ex: peer1)
  qr <peer>          Affiche le QR code d'un client
  regenerate         Régénère toutes les configs clients
  logs               Affiche les logs en temps réel
  test               Test de connectivité

Examples:
  $0 start                 # Démarre WireGuard
  $0 show-config peer1     # Affiche la config du client 1
  $0 qr peer1              # Affiche le QR code du client 1
  $0 status                # Vérifie l'état du VPN

EOF
}

# Test de connectivité
test_connectivity() {
    print_info "Test de connectivité...\n"

    # Vérifier que le container tourne
    if ! docker ps | grep -q signflow_wireguard; then
        print_error "WireGuard n'est pas démarré"
        exit 1
    fi
    print_success "Container WireGuard actif"

    # Vérifier que le port UDP est ouvert
    print_info "Vérification du port UDP 51820..."
    if command -v nc &> /dev/null; then
        # Note : nc ne peut pas vraiment tester UDP, mais on peut vérifier l'écoute
        if docker exec signflow_wireguard ss -ulnp | grep -q 51820; then
            print_success "Port UDP 51820 en écoute"
        else
            print_error "Port UDP 51820 non accessible"
        fi
    else
        print_warning "Outil 'nc' non disponible, skip test port UDP"
    fi

    # Afficher l'IP publique
    print_info "Détection de l'IP publique..."
    PUBLIC_IP=$(curl -s -4 ifconfig.me || echo "Inconnu")
    echo "   IP publique détectée : $PUBLIC_IP"
    echo "   Les clients doivent se connecter à : $PUBLIC_IP:51820"

    echo ""
    print_success "Tests terminés"
}

# Main
main() {
    print_header

    case "${1:-}" in
        start)
            check_requirements
            start_vpn
            ;;
        stop)
            stop_vpn
            ;;
        restart)
            stop_vpn
            sleep 2
            start_vpn
            ;;
        status)
            show_status
            ;;
        list)
            show_clients
            ;;
        show-config)
            show_client_config "$2"
            ;;
        qr)
            show_qr_code "$2"
            ;;
        regenerate)
            regenerate_configs
            ;;
        logs)
            print_info "Logs WireGuard (Ctrl+C pour quitter) :\n"
            docker logs -f signflow_wireguard
            ;;
        test)
            test_connectivity
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Commande inconnue : ${1:-}"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

main "$@"
