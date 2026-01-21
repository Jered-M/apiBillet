#!/usr/bin/env node

/**
 * Diagnostic: Tester le modèle avec une image 500 CDF vs 20 USD
 * 
 * Usage:
 * node test-model-confusion.js path/to/image.jpg
 */

const fs = require('fs');
const path = require('path');

const API_URL = 'https://apibillet-1.onrender.com/predict';

async function testImage(imagePath) {
  if (!fs.existsSync(imagePath)) {
    console.error(`❌ Fichier non trouvé: ${imagePath}`);
    process.exit(1);
  }

  console.log('\n🧪 Test Confusion 500 CDF vs 20 USD');
  console.log('═══════════════════════════════════\n');
  console.log(`📸 Image: ${path.basename(imagePath)}`);
  console.log(`📊 Taille: ${(fs.statSync(imagePath).size / 1024).toFixed(2)} KB\n`);

  try {
    // Créer FormData
    const FormData = require('form-data');
    const formData = new FormData();
    formData.append('file', fs.createReadStream(imagePath));

    console.log('📤 Envoi vers API...');
    const response = await fetch(API_URL, {
      method: 'POST',
      body: formData,
      headers: formData.getHeaders(),
    });

    console.log(`📨 Réponse: HTTP ${response.status}\n`);

    if (!response.ok) {
      const error = await response.text();
      console.error(`❌ Erreur: ${error}`);
      process.exit(1);
    }

    const data = await response.json();

    console.log('✅ Résultat:');
    console.log(`   Classe: ${data.result}`);
    console.log(`   Confiance: ${(data.confidence * 100).toFixed(2)}%\n`);

    if (data.all_scores) {
      console.log('📊 Toutes les prédictions:');
      
      // Mapper index → label
      const labels = {
        0: "100 CDF",
        1: "50 CDF",
        2: "200 CDF",
        3: "500 CDF",
        4: "1000 CDF",
        5: "5000 CDF",
        6: "10000 CDF",
        7: "20000 CDF",
        8: "100 USD",
        9: "5 USD",
        10: "10 USD",
        11: "50 USD",
        12: "20 USD",
        13: "1 USD",
      };

      // Créer array avec scores
      const scores = Object.entries(data.all_scores)
        .map(([idx, score]) => ({
          index: parseInt(idx),
          label: labels[parseInt(idx)] || `Unknown ${idx}`,
          score: parseFloat(score),
        }))
        .sort((a, b) => b.score - a.score);

      // Afficher top 5
      scores.slice(0, 5).forEach((item, idx) => {
        const bar = '█'.repeat(Math.round(item.score * 20));
        const pct = (item.score * 100).toFixed(2);
        console.log(`   ${idx + 1}. ${item.label.padEnd(12)} ${bar.padEnd(20)} ${pct}%`);
      });

      console.log('\n🔍 Confusion détectée?');
      const top2 = scores.slice(0, 2);
      if ((top2[0].label === '500 CDF' && top2[1].label === '20 USD') ||
          (top2[0].label === '20 USD' && top2[1].label === '500 CDF')) {
        console.log(`   ⚠️  ${top2[0].label} vs ${top2[1].label} très proches!`);
        console.log(`       Différence: ${Math.abs(top2[0].score - top2[1].score).toFixed(4)}`);
        
        // Recommandation
        console.log('\n💡 Recommandations:');
        console.log('   1. Vérifier que le modèle a assez de données pour ces deux classes');
        console.log('   2. Augmenter le dataset avec des images 500 CDF et 20 USD');
        console.log('   3. Vérifier que le preprocessing est identique à Colab');
        console.log('   4. Tester le même modèle.h5 depuis Colab ici');
      } else {
        console.log('   ✅ Pas de confusion détectée');
      }
    }
  } catch (error) {
    console.error(`\n❌ Erreur: ${error.message}`);
    process.exit(1);
  }
}

// Argument: chemin image
const imagePath = process.argv[2];
if (!imagePath) {
  console.error('Usage: node test-model-confusion.js <path-to-image>');
  console.error('Example: node test-model-confusion.js test_bills/500-cdf.jpg');
  process.exit(1);
}

testImage(imagePath);
